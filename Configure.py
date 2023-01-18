# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
	"modeldir": 'saved_models',
	"layers": 190,
	"growthrate": 40,
	# "layers": 100,
	# "growthrate": 12,
	"compression_factor": 0.5,
	"bottleneck": True,
	"nclasses": 10,
	# "learning_rate": 0.01,
	# "weight_decay": 2e-4,
	"learning_rate": 0.1,
	"weight_decay": 1e-4,
	"batch_size": 32,
	#"batch_size": 64,
	"save_interval": 10,
	"momentum": 0.9
}

training_configs = {
	# "learning_rate": 0.01,
	# "max_epoch": 300
	# "max_epoch": 150
	"max_epoch": 200
	# ...
}

### END CODE HERE