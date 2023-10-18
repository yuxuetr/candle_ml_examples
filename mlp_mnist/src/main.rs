use clap::{ Parser };
use candle_core::{ Device, Result, Tensor, D, DType };
use candle_nn::{ loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap };


const IMAGE_DIM: usize = 784;
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

struct MLP {
    ln1: Linear,
    ln2: Linear,
}

trait Model: Sized {
    fn new(vs: VarBuilder) -> Result<Self>;
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

impl Model for MLP {
    fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(IMAGE_DIM, 100, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(100, LABELS, vs.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        self.ln2.forward(&xs)
    }
}

fn train<M: Model>(
    m: candle_datasets::vision::Dataset,
    args: &TrainingArgs) -> anyhow::Result<()> {

    let dev = Device::Cpu;

    let train_labels = m.train_labels;
    let train_images = m.train_images.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let test_images = m.test_images.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = M::new(vs.clone())?;

    // Load Pre-trained Model Parameters
    if let Some(load_path) = &args.load_path {
        println!("Loading model from {}", load_path);
        let _ = varmap.load(load_path);
    }

    // Create Optimizer
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), args.learning_rate)?;

    // Iterate training model
    for epoch in 1..=args.epochs {
        let start_time = std::time::Instant::now();
        let logits = model.forward(&train_images)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_labels)?;

        sgd.backward_step(&loss)?;

        let test_logits = model.forward(&test_images)?;
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
                 loss.to_scalar::<f32>()?, test_accuracy * 100., epoch_duration.as_secs_f64());
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

    let default_learning_rate: f64 = 0.05;

    let training_args = TrainingArgs {
        epochs: args.epochs,
        learning_rate: args.learning_rate.unwrap_or(default_learning_rate),
        load_path: args.load_model_path,
        save_path: args.save_model_path,
    };

    train::<MLP>(m, &training_args)
}
