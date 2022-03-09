from dataclasses import dataclass


@dataclass
class BaseOptions:
	channels_out: int = 3
	height: int = 256
	width: int = 256
	number_of_classes: int = 2
	channels_in: int = 3

	learning_rate: float = 0.001
	batch_size: int = 1

	number_of_epochs: int = 200
	epoch_between_decay: int = 50
	decay_rate: float = 0.5

	dataloader_threads: int = 70
	device: str = "cuda"
	verbose: bool = True

	background_label: int = 0