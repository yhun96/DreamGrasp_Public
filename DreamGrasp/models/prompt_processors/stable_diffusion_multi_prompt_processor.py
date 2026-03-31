import threestudio
from threestudio.utils.base import BaseObject
from dataclasses import dataclass, field
from copy import deepcopy
import ast

@threestudio.register("stable-diffusion-multi-prompt-processor")
class StableDiffusionMultiPromptProcessor(BaseObject):
    @dataclass
    class Config:
        pretrained_model_name_or_path: str = ""
        prompts: list = field(default_factory=lambda: [])
        prompt_path : str = ""
        prefix: str = ""
        prompt: str = ""
    
    cfg: Config
    
    def configure(self):
        if self.cfg.prompt_path != "":
            if len(self.cfg.prompts) > 0:
                raise NotImplementedError("prompt and prompt path are not compatible")
            else:
                with open(self.cfg.prompt_path, "r") as text_file:
                    prompts = ast.literal_eval(text_file.readline())
        else:
            prompts = self.cfg.prompts
        del self.cfg.prompt_path
        del self.cfg.prompts
        self.prompt_utils = []
        for prompt in prompts:
            cfg = deepcopy(self.cfg)
            cfg.prompt = cfg.prefix + prompt
            del cfg.prefix
            self.prompt_processor = threestudio.find("stable-diffusion-prompt-processor")(cfg)
            self.prompt_utils.append(self.prompt_processor())
            
    def __call__(self):
        return self.prompt_utils
        
            