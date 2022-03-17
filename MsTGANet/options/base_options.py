from dataclasses import dataclass


@dataclass
class BaseOptions:
	channels: int
	height: int
	width: int
	number_of_classes: int

	learning_rate: float = 0.001
	batch_size: int = 10

	number_of_epochs: int = 200
	epoch_between_decay: int = 50
	decay_rate: float = 0.5

	dataloader_threads: int = 1
	device: str = "cpu"
	verbose: bool = True

	background_label: int = 0