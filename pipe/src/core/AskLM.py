import logging

from vllm import SamplingParams
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        dataset,
        instruction,
        max_context_length,
        max_new_tokens,
        chat_model,
        sys_prompt,
        few_shots=None,
        prompt_subs=None,
        truncate=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.instruction = instruction
        self.max_context_length = max_context_length
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.chat_model = chat_model
        self.sys_prompt = sys_prompt
        self.few_shots = few_shots
        self.truncate = truncate
        if prompt_subs is None:
            self.prompt_subs = {
                'context': 'text',
                'input': 'question',
                'sample_answer': 'answer'
            }
        else:
            self.prompt_subs = prompt_subs

    def _create_prompt(self, sample):
        prompt = self.instruction
        for pattern, column in self.prompt_subs.items():
            if column in sample:
                prompt = prompt.replace(
                    '{' + pattern + '}',
                    str(sample[column])
                )
        return prompt

    def create_prompt(self, sample):
        return self._create_prompt(sample)

    def truncate_text(self, text, max_new_len):
        tokenized_text = self.tokenizer(
            text,
            truncation=True,
            max_length=max_new_len,
            add_special_tokens=False)['input_ids']
        truncated_text = self.tokenizer.decode(tokenized_text)

        return truncated_text

    def create_chat_messages(self, sample):
        messages = []
        if self.sys_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": self.sys_prompt,
                }
            )
        if self.few_shots:
            messages.extend(self.few_shots)

        if self.truncate:
            truncate_column = self.truncate
            tokenized_messages_length = len(self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )) if len(messages) else 0

            max_prompt_length = self.max_context_length - tokenized_messages_length
            sample[truncate_column] = self.truncate_text(
                text=sample[truncate_column],
                max_new_len=int(max_prompt_length*0.9))

        prompt = self._create_prompt(sample)
        messages.append({"role": "user", "content": prompt})
        return messages

    def create_prompt_with_chat_template(self, sample):
        messages = self.create_chat_messages(sample)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=True,
            max_length=self.max_context_length,
        )

        return prompt

    def generate_answers(self):
        generated_answers = []

        for sample in tqdm(self.dataset):
            if self.chat_model:
                if (
                    not hasattr(self.tokenizer, "chat_template")
                    or not self.tokenizer.chat_template
                ):
                    logger.info(
                        "chat_model is set to True, "
                        "but tokenizer doesn't have chat_template. "
                        "Evaluating w/o template."
                    )
                    prompt = self.create_prompt(sample)
                else:
                    prompt = self.create_prompt_with_chat_template(sample)
            else:
                prompt = self.create_prompt(sample)

            inputs = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_context_length,
                return_tensors="pt",
            ).to(self.device)

            generation_output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id
            )

            prompt_len = len(inputs["input_ids"][0])

            model_answer = self.tokenizer.decode(
                generation_output[0][prompt_len:].cpu(),
                skip_special_tokens=True
            )
            generated_answer = {
                k: sample[v] for k, v in self.prompt_subs.items()
            } | {'model_answer': model_answer}
            generated_answers.append(generated_answer)
        return generated_answers


class vLLM_AnswerGenerator(AnswerGenerator):
    def create_prompt(self, sample):
        prompt = super().create_prompt(sample)
        sample["prompt"] = prompt
        return sample

    def create_prompt_with_chat_template(self, sample):
        prompt = super().create_prompt_with_chat_template(sample)
        sample["prompt"] = prompt
        return sample

    def generate_answers(self):
        if self.chat_model:
            self.dataset = self.dataset.map(
                self.create_prompt_with_chat_template
            )
        else:
            self.dataset = self.dataset.map(self.create_prompt)

        generated_answers = []

        if len(self.dataset) == 0:
            return generated_answers

        out = self.model.generate(
            self.dataset["prompt"],
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=self.max_new_tokens,
                truncate_prompt_tokens=self.max_context_length,
            ),
        )

        for ind, request in enumerate(out):
            model_answer = request.outputs[0].text
            # prompt = request.prompt
            generated_answer = {
                k: self.dataset[ind][v]
                for k, v in self.prompt_subs.items()
                if v in self.dataset.features
            } | {
                'model_answer': model_answer,
                # 'prompt': prompt
            }
            generated_answers.append(generated_answer)
        return generated_answers


class API_AnswerGenerator(AnswerGenerator):
    def __init__(
        self,
        api,
        model_name,
        dataset,
        instruction,
        max_context_length,
        max_new_tokens,
        chat_model,
        sys_prompt,
        few_shots=None,
        prompt_subs=None
    ):
        self.api = api
        self.model_name = model_name
        self.dataset = dataset
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens
        self.instruction = instruction
        self.chat_model = chat_model
        self.sys_prompt = sys_prompt
        self.few_shots = few_shots
        if prompt_subs is None:
            self.prompt_subs = {
                'context': 'text',
                'input': 'question',
                'sample_answer': 'answer'
            }
        else:
            self.prompt_subs = prompt_subs

    def generate_answers(self):
        generated_answers = []

        for sample in tqdm(self.dataset):
            if self.chat_model:
                messages = self.create_chat_messages(sample)
                try:
                    response = self.api.chat.completions.create(
                        model=self.model_name,
                        messages=messages
                    )
                    model_answer = response.choices[0].message.content
                except Exception as e:
                    logger.exception(e)
                    model_answer = ''
            else:
                prompt = self.create_prompt(sample)
                response = self.api.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=self.max_new_tokens
                )
                model_answer = response.choices[0].text

            generated_answer = {
                k: sample[v] for k, v in self.prompt_subs.items()
            } | {'model_answer': model_answer}
            generated_answers.append(generated_answer)
        return generated_answers
