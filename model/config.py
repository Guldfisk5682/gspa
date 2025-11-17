from transformers import PretrainedConfig


class GSPAConfig(PretrainedConfig):
    model_type = "gspa"

    def __init__(
        self,
        model_name_or_path="openai/clip-vit-base-patch16",
        ctx=4,
        class_names=[],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path
        self.ctx = ctx
        self.class_names = class_names
