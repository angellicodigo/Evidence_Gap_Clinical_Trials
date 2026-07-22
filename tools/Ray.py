from pathlib import Path
from typing import Any
import ray
from ray.util import ActorPool
from ray.actor import ActorHandle
from vllm import LLM, SamplingParams
import socket
from ray.runtime_context import get_runtime_context

# Put your own path for SCRATCH_PATH
SCRATCH_PATH = Path("/sc/arion/scratch/lia38")
ADDRESS = (SCRATCH_PATH / "ray_address").read_text().strip()

def init_ray() -> None:
    if not ray.is_initialized():
        ray.init(address=ADDRESS, namespace="nemotron")

def get_cluster_resources(debug: bool = False) -> dict[str, float]:
    init_ray()
    if debug:
        for n in ray.nodes():
            print("NodeID:", n["NodeID"])
            print("IP:", n["NodeManagerAddress"])
            print("Alive:", n["Alive"])
            print("Resources:", n["Resources"])
            print("-" * 60)
    return ray.cluster_resources()

@ray.remote(num_gpus=1, resources={"b200": 0.01})
class NemotronB200:
    def __init__(self, name: str):
        self.hostname = socket.gethostname()
        self.node_id = get_runtime_context().get_node_id()
        self.name = name
        print(f"Actor: {self.name}")
        print(f"Host: {self.hostname}")
        print(f"Node ID: {self.node_id}")
        self.model = LLM(
            model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
            tensor_parallel_size=1,
            dtype="auto",
            kv_cache_dtype="fp8",
            gpu_memory_utilization=0.90,
            max_model_len=1048576,
        )

    def chat(self, messages: list[list[dict[str, Any]]], sampling_params: SamplingParams, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if tools is None:
            result = self.model.chat(
                messages,
                sampling_params=sampling_params,
            )
        else:
            result = self.model.chat(
                messages,
                sampling_params=sampling_params,
                tools=tools,
            )

        return {
            "name": self.name,
            "hostname": self.hostname,
            "node_id": self.node_id,
            "result": result
        }


@ray.remote(num_gpus=4, resources={"l40": 0.01})
class NemotronL40:
    def __init__(self, name: str):
        self.hostname = socket.gethostname()
        self.node_id = get_runtime_context().get_node_id()
        self.name = name
        print(f"Actor: {self.name}")
        print(f"Host: {self.hostname}")
        print(f"Node ID: {self.node_id}")
        self.model = LLM(
            model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
            tensor_parallel_size=4,
            dtype="auto",
            kv_cache_dtype="fp8",
            gpu_memory_utilization=0.90,
            max_model_len=1048576,
        )

    def chat(self, messages: list[list[dict[str, Any]]], sampling_params: SamplingParams, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if tools is None:
            result = self.model.chat(
                messages,
                sampling_params=sampling_params,
            )
        else:
            result = self.model.chat(
                messages,
                sampling_params=sampling_params,
                tools=tools,
            )

        return {
            "name": self.name,
            "hostname": self.hostname,
            "node_id": self.node_id,
            "result": result
        }

def get_actor_pool() -> ActorPool:

    def __retrieve_actor__(actor_cls: Any, name: str) -> ActorHandle:
        try:
            actor = ray.get_actor(name)
            print(f"Reusing actor: {name}")
            return actor
        except ValueError:
            print(f"Creating actor: {name}")
            return actor_cls.options(
                name=name,
                lifetime="detached",
            ).remote(name)

    init_ray()
    resources = ray.cluster_resources()
    actors = []
    num_b200 = int(resources.get("b200", 0))
    for i in range(num_b200):
        actors.append(
            __retrieve_actor__(
                NemotronB200,
                f"b200-{i}",
            )
        )

    num_l40 = int(resources.get("l40", 0))
    for i in range(num_l40):
        actors.append(
            __retrieve_actor__(
                NemotronL40,
                f"l40-{i}",
            )
        )

    if not actors:
        raise RuntimeError("No Nemotron workers found.")

    return ActorPool(actors)