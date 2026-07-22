from pathlib import Path
from ray.util import ActorPool
from Evidence_Gap_Clinical_Trials.tools.Ray import get_actor_pool
from Evidence_Gap_Clinical_Trials.tools.tasks import initialize_drugs, run_initial_queries, evaluate_trials, save_results
from vllm import SamplingParams


TRIAL_SOURCE_PATH = Path("/sc/arion/projects/EHR_ML/lia38/nemotronL40/data/clinical_trials/")

def find_relevant_trials(model: ActorPool, sampling_params: SamplingParams, drugs_path: Path, batch_size: int = 256, max_iterations: int = 2) -> None: 
    output_dir = drugs_path / "trials"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    drug_states, class_data_results, class_reasoning_logs, initial_prompts_queue = initialize_drugs(drugs_path)
    run_initial_queries(model, sampling_params, drug_states, initial_prompts_queue, batch_size, TRIAL_SOURCE_PATH)
    evaluate_trials(model, sampling_params, drug_states, class_data_results, class_reasoning_logs, batch_size, max_iterations, TRIAL_SOURCE_PATH)
    save_results(output_dir, class_data_results, class_reasoning_logs)

if __name__ == '__main__':
    drugs_path = Path("/sc/arion/projects/EHR_ML/lia38/nemotronL40/data/responses/DISEASE")
    pool = get_actor_pool()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100000)
    find_relevant_trials(pool, sampling_params, drugs_path)