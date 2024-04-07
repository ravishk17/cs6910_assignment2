# cs6910_assignment2
|Libraries required|
|------------------|
| cv2 |
| glob |
| random |
| torch |
| torchvision |
| numpy |
| matplotlib.pyplot |
| pandas.core.common |
| tqdm |
| PIL |
| wandb |

**Instructions to run the program for partA:**

1)Unzip the iNaturalist dataset in the location where trainA.py is located. That is the trainA.py and 'inaturalist_12K' folder is at the same level. The inner structure of the dataset need not be modified or renamed, otherwise path mentioned inside the code will be wrong.

2)Below are the command line arguments which can be given at runtime
| Name | Default Value | Description |
|------|---------------|-------------|
| -wp, --wandb_project | assignement2_kaggle | Project name used to track experiments in Weights & Biases dashboard |
| -we, --wandb_entity | cs23m055 | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| -bn, --batch_norm | True | choices: [True, False] |
| -ndl, --neurons_in_dense_layer | 1024 | number of neurons in the dense layer|
| -e, --epochs | 5 | number of epochs |
| -b, --batch_size | 32 | Batch size used in the training |
| -aug, --data_aug | False | choices: [True, False] |
| -fo, --filter_organization | 1 | choices: [1, 2, 0.5] |
| -nif, --number_initial_filters | 128 | number initial filters |
| -a, --activation | 'mish' | activation function |
| -lr, --learning_rate | 0.00001 | learning rate used to optimize the model |
| -d, --dropout | 0.2 | Dropout to take care of overfitting |
| -sf, --size_filters| [7,5,5,3,3] | Give exactly 5 space separated integers after -sf to use this command|

To run the partA: ``` python trainA.py -e 5 -sf 5 5 5 5 5``` , ```python trainA.py```, and so on try with different commandline arguments

**Instructions to execute partB: **

1)Unzip the iNaturalist dataset in the location where trainB.py is located. That is the trainB.py and 'inaturalist_12K' folder is at the same level. The inner structure of the dataset need not be modified or renamed, otherwise path mentioned inside the code will be wrong.

2)Below are the command line arguments which can be given at runtime

| Name | Default Value | Description |
|------|---------------|-------------|
| -e, --epochs | 5 | number of epochs |
| -s', --strategy | 1 | choices = [0:freeze_all, 1:fine_tuning_all, 2:layer_wise_fine_tuning] |

To run the the partB: ```python trainB.py -e 5 -s 1```


