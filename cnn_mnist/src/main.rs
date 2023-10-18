use clap::{ Parser };
use rand::prelude::*;
use candle_core::{ Device, Result, Tensor, D, DType };
use candle_nn::{ loss, ops, Linear, Conv2d, Dropout, Optimizer, VarBuilder, VarMap };


const LABELS: usize = 10;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    learning_rate: Option<f64>,

    #[arg(long, default_value_t = 10)]
    epochs: usize,

    #[arg(long)]
    save_model_path: Option<String>,

    #[arg(long)]
    load_model_path: Option<String>,

    #[arg(long)]
    local_mnist: Option<String>,
}

struct TrainingArgs {
    learning_rate: f64,
    load_path: Option<String>,
    save_path: Option<String>,
    epochs: usize,
}

#[derive(Debug)]
struct LeNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    dropout: Dropout,
}

impl LeNet {
    fn new(vs: VarBuilder) -> Result<Self> {
        let conv1 = candle_nn::conv2d(1, 32, 5, Default::default(), vs.pp("conv1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 5, Default::default(), vs.pp("conv2"))?;
        let fc1 = candle_nn::linear(1024, 1024, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(1024, LABELS, vs.pp("fc2"))?;
        let dropout = Dropout::new(0.5);
        Ok(Self { conv1, conv2, fc1, fc2, dropout })
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (b_sz, _img_dim) = xs.dims2()?;
        let xs = xs
            .reshape((b_sz, 1, 28, 28))?
            .apply(&self.conv1)?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .max_pool2d(2)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            .relu()?;
        self.dropout.forward(&xs, train)?.apply(&self.fc2)
    }
}


fn train(
    m: candle_datasets::vision::Dataset,
    args: &TrainingArgs) -> anyhow::Result<()> {

    const BSIZE: usize = 16;
    let dev = Device::Cpu;

    let train_labels = m.train_labels;
    let train_images = m.train_images.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let test_images = m.test_images.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = LeNet::new(vs.clone())?;

    // Load Pre-trained Model Parameters
    if let Some(load_path) = &args.load_path {
        println!("Loading model from {}", load_path);
        let _ = varmap.load(load_path);
    }

    // Create Optimizer
    let adamw_params = candle_nn::ParamsAdamW {
        lr: args.learning_rate,
        ..Default::default()
    };

    let mut opt = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;

    // Batch
    let n_batches = train_images.dim(0)? / BSIZE;
    let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();

    // Iterate training model
    for epoch in 1..=args.epochs {
        let mut sum_loss = 0f32;
        batch_idxs.shuffle(&mut rand::thread_rng());
        let start_time = std::time::Instant::now();

        for batch_idx in batch_idxs.iter() {
            let train_images = train_images.narrow(0, batch_idx * BSIZE, BSIZE)?;
            let train_labels = train_labels.narrow(0, batch_idx * BSIZE, BSIZE)?;
            let logits = model.forward(&train_images, true)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_sm, &train_labels)?;
            opt.backward_step(&loss)?;
            sum_loss += loss.to_vec0::<f32>()?;
        }

        let avg_loss = sum_loss / n_batches as f32;

        let test_logits = model.forward(&test_images, false)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;

        let end_time = std::time::Instant::now();
        let epoch_duration = end_time.duration_since(start_time);
        println!("Epoch: {epoch:4} Train Loss: {:8.5} Test Acc: {:5.2}% Epoch duration: {:.2} second.",
                 avg_loss, test_accuracy * 100., epoch_duration.as_secs_f64());
    }

    // Save Model Parameters
    if let Some(save_path) = &args.save_path {
        println!("Saving trained weight in {save_path}");
        varmap.save(save_path)?
    }
    Ok(())
}


fn main() ->anyhow::Result<()> {
    let args: Args = Args::parse();
    let m: candle_datasets::vision::Dataset = if let Some(directory) = args.local_mnist {
        candle_datasets::vision::mnist::load_dir(directory)?
    } else {
        candle_datasets::vision::mnist::load()?
    };

    println!("Train Images: {:?}", m.train_images.shape());
    println!("Train Labels: {:?}", m.train_labels.shape());
    println!("Test  Images: {:?}", m.test_images.shape());
    println!("Test  Labels: {:?}", m.test_labels.shape());

    let default_learning_rate: f64 = 0.001;

    let training_args = TrainingArgs {
        epochs: args.epochs,
        learning_rate: args.learning_rate.unwrap_or(default_learning_rate),
        load_path: args.load_model_path,
        save_path: args.save_model_path,
    };

    train(m, &training_args)
}
