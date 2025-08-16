from .phosphobot import PhosphoClient # type: ignore

def launch_replay(episode_id: int, dataset_name: str, phospho: PhosphoClient):
    phospho.post("/recording/play", json={
        "dataset_name": dataset_name,
        "episode_id": episode_id,
    })
