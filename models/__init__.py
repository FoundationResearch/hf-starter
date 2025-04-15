from .llama3 import MyLlamaConfig, MyLlamaModel, MyLlamaForCausalLM
from .prob_t import ProbTConfig, ProbTModel, ProbTForCausalLM

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForTokenClassification

# 注册llama3模型
AutoConfig.register("my-llama", MyLlamaConfig)
AutoModel.register(MyLlamaConfig, MyLlamaModel)
AutoModelForCausalLM.register(MyLlamaConfig, MyLlamaForCausalLM)

# 注册prob-t模型
AutoConfig.register("prob-t", ProbTConfig)
AutoModel.register(ProbTConfig, ProbTModel)
AutoModelForCausalLM.register(ProbTConfig, ProbTForCausalLM)
