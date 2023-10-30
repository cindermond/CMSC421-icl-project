# CMSC421-icl-project
This is the codebase for UMD CMSC421 icl project. We provide methods to do in-context learning on two datasets---sst2 and gsm8k with Llama-2 chat version.

## Important Note
By default you only have 20G of disk space, which is definitely not enough for Llama-2-13B. Thus, you need to read the code first. Start with the main python files you run, i.e., the ones in the `run` folder. The variable `prompt_dict` is imported from `batcon/consts.py`. You can easily locate the other imported classes. Read these classes. All the namings are straightforward. After you read the code and have a plan about where to modify, let me know in the Discord channel. If you want to get a sense of what the code is doing, replace the default model with the 7B version (I haven't tested this but I think the space should be tightly enough.) and follow the instructions.

## Apply for Llama-2 access
To use Llama-2, you need to first apply for access on [the official website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/). After you submit the form, you will soon get an email from Meta titled "Get started with Llama 2". This means that you are granted access to Llama-2 for 24 hours. The next step is to fill in the application form on [HuggingFace](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf). I'm not sure if you can still do this after the 24 hours limit. Wait for some time and your HuggingFace account will be granted Llama-2 access. After that, find your HuggingFace access token in [your account setting](https://huggingface.co/settings/tokens). Copy the token and paste it to `configs/huggingface.token`.

## Set up the environment
In the folder that contains this readme file, run 
```
pip install -e .
```

## Run the code
To test the code on the first 2 examples of sst2, run 
```
python run/few_shot_glue.py --valid_limit 2 --verbose
```
I suggest that you redirect the output to a text file like
```
python run/few_shot_glue.py --valid_limit 2 --verbose > output_sst2.out
```
To test the code on the first 2 examples of gsm8k, run
```
python run/few_shot_gsm.py --valid_limit 2 --verbose > output_gsm.out
```

Check the output files to get a sense of what the code is doing.

