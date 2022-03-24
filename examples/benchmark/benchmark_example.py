from optimum_transformers import Benchmark

task = "sentiment-analysis"
model_name = "philschmid/MiniLM-L6-H384-uncased-sst2"
num_tests = 100

benchmark = Benchmark(task, model_name)
results = benchmark(num_tests, plot=True)
