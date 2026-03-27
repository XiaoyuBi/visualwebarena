import argparse
import json
import re
from typing import Any, Optional

import tiktoken
from beartype import beartype
from PIL import Image

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    ActionTypes,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from browser_env.utils import Observation, StateInfo, pil_to_b64
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)
from llms.tokenizers import Tokenizer


class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError


class TeacherForcingAgent(Agent):
    """Agent that follows a pre-defined action sequence"""

    def __init__(self) -> None:
        super().__init__()

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def set_actions(self, action_seq: str | list[str]) -> None:
        if isinstance(action_seq, str):
            action_strs = action_seq.strip().split("\n")
        else:
            action_strs = action_seq
        action_strs = [a.strip() for a in action_strs]

        actions = []
        for a_str in action_strs:
            try:
                if self.action_set_tag == "playwright":
                    cur_action = create_playwright_action(a_str)
                elif self.action_set_tag in [
                    "id_accessibility_tree",
                    "id_accessibility_tree_with_captioner",
                ]:
                    cur_action = create_id_based_action(a_str)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
            except ActionParsingError as e:
                cur_action = create_none_action()

            cur_action["raw_prediction"] = a_str
            actions.append(cur_action)

        self.actions: list[Action] = actions

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        return self.actions.pop(0)

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        with open(test_config_file) as f:
            ref_actions = json.load(f)["reference_action_sequence"]
            tag = ref_actions["action_set_tag"]
            action_seq = ref_actions["action_sequence"]
            self.set_action_set_tag(tag)
            self.set_actions(action_seq)


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
        captioning_fn = None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn

        # Check if the model is multimodal (vision-capable).
        model_lower = lm_config.model.lower()
        is_vision_model = (
            "gemini" in model_lower
            or ("gpt-4" in model_lower and ("vision" in model_lower or "gpt-4o" in model_lower))
            or ("qwen" in model_lower and "vl" in model_lower)
            or "gpt-5" in model_lower
        )
        if is_vision_model and type(prompt_constructor) == MultimodalCoTPromptConstructor:
            self.multimodal_inputs = True
        else:
            self.multimodal_inputs = False

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @beartype
    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any], images: Optional[list[Image.Image]] = None,
        output_response: bool = False
    ) -> Action:
        # Create page screenshot image for multimodal models.
        if self.multimodal_inputs:
            page_screenshot_arr = trajectory[-1]["observation"]["image"]
            page_screenshot_img = Image.fromarray(
                page_screenshot_arr
            )  # size = (viewport_width, viewport_width)

        # Caption the input image, if provided.
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                image_input_caption = ""
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                # Update intent to include captions of input images.
                intent = f"{image_input_caption}\nIntent: {intent}"
            elif not self.multimodal_inputs:
                raise ValueError(
                    "Input image provided but no image captioner available. "
                    "Cannot process task - early stopping."
                )

        if self.multimodal_inputs:
            prompt = self.prompt_constructor.construct(
                trajectory, intent, page_screenshot_img, images or [], meta_data
            )
        else:
            prompt = self.prompt_constructor.construct(
                trajectory, intent, meta_data
            )
        lm_config = self.lm_config
        n = 0
        while True:
            response = call_llm(lm_config, prompt)
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            if output_response:
                print(f'Agent: {response}', flush=True)
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(
                    response
                )
                if self.action_set_tag in [
                    "id_accessibility_tree",
                    "id_accessibility_tree_with_captioner",
                ]:
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
                action["raw_prediction"] = response
                break
            except ActionParsingError as e:
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    break

        return action

    @beartype
    def evaluate_stop_decision(
        self,
        action: Action,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        images: Optional[list[Image.Image]] = None,
    ) -> Action:
        """LLM second-pass: verify the stop decision and optionally override it.

        - If action is STOP: confirm the stop is correct; if not, return a
          replacement action to continue with.
        - If action is non-STOP: check whether the task is already complete;
          if so, return a STOP action with the answer.

        Falls back to the original action on any LLM or parse failure.
        """
        state_info = trajectory[-1]  # type: ignore[index]
        obs: str = state_info["observation"]["text"]  # type: ignore[index]

        # Truncate observation to token budget.
        max_obs_length = self.lm_config.gen_config.get("max_obs_length", 0)
        if max_obs_length:
            if self.lm_config.provider == "google":
                obs = obs[:max_obs_length]
            else:
                obs = self.prompt_constructor.tokenizer.decode(
                    self.prompt_constructor.tokenizer.encode(obs)[:max_obs_length]  # type: ignore[arg-type]
                )

        url: str = state_info["info"]["page"].url  # type: ignore[index]

        # Full numbered action history for richer context.
        action_history_str = "\n".join(
            f"{i + 1}. {a}" for i, a in enumerate(meta_data["action_history"])
        )

        is_stop = action["action_type"] == ActionTypes.STOP

        if is_stop:
            system_prompt = (
                "You are a critical evaluator for a web automation agent. "
                "The agent has decided to STOP and submit a final answer.\n"
                "Your job: verify whether this stop decision is truly correct given the current page state.\n\n"
                "Reason step-by-step through the following:\n"
                "1. What exactly does the objective require? "
                "(a specific piece of information, a completed action, a navigation target, etc.)\n"
                "2. Is the required information or confirmation of completion clearly visible "
                "in the current observation?\n"
                "3. Is the agent's proposed answer accurate and complete based on what is shown?\n"
                "4. Does the action history show the agent actually reached the right state "
                "(e.g., navigated to the correct page, submitted the right form)?\n\n"
                "Output format — reply with your reasoning, then end with EXACTLY ONE of:\n"
                "- ```keep``` if stopping is correct and the answer is accurate.\n"
                "- ```<next_action>``` (e.g. ```click [42]```, ```scroll [down]```, "
                "```goto [url]```) if the task is NOT yet complete. "
                "Choose the most logical next step based on the current observation."
            )
            user_message = (
                f"OBJECTIVE: {intent}\n\n"
                f"ACTION HISTORY (all steps taken so far):\n{action_history_str}\n\n"
                f"CURRENT URL: {url}\n\n"
                f"CURRENT OBSERVATION:\n{obs}\n\n"
                f"AGENT'S PROPOSED STOP ANSWER: {action['answer']}\n\n"
                "Is stopping correct here? Reason step-by-step, then reply with "
                "```keep``` or the next action to take."
            )
        else:
            system_prompt = (
                "You are a critical evaluator for a web automation agent. "
                "The agent is about to take another action, but you must first check "
                "whether the task is already complete.\n"
                "Your job: determine if the current page already contains sufficient "
                "information to answer the objective — without any further actions.\n\n"
                "Reason step-by-step through the following:\n"
                "1. What exactly does the objective require? "
                "(find specific information, confirm an action was done, etc.)\n"
                "2. Carefully scan the current observation: is the required answer or "
                "completion evidence already present on the page?\n"
                "3. Review the action history: has the agent already done what was needed?\n"
                "4. Would taking the agent's proposed next action actually help, "
                "or is it unnecessary?\n\n"
                "Output format — reply with your reasoning, then end with EXACTLY ONE of:\n"
                "- ```keep``` if the task is not yet complete and the agent should continue.\n"
                "- ```stop [answer]``` if the task IS already complete. "
                "The answer must be the exact value the objective asked for "
                '(e.g., a price like "$12.99", a name, a count, "N/A", '
                'or "done" for action tasks).'
            )
            user_message = (
                f"OBJECTIVE: {intent}\n\n"
                f"ACTION HISTORY (all steps taken so far):\n{action_history_str}\n\n"
                f"CURRENT URL: {url}\n\n"
                f"CURRENT OBSERVATION:\n{obs}\n\n"
                f"AGENT'S PROPOSED NEXT ACTION: {action.get('raw_prediction', str(action))}\n\n"
                "Is the task already complete? Reason step-by-step, then reply with "
                "```keep``` or ```stop [answer]```."
            )

        # Build provider-specific API input.
        # Qwen2.5 models (Qwen2.5-VL-*) are served via an OpenAI-compatible
        # endpoint (Hyperbolic / OpenRouter) with provider="openai" and are
        # covered by the first branch below.
        if "openai" in self.lm_config.provider and self.lm_config.mode == "chat":
            if self.multimodal_inputs:
                page_screenshot_arr = state_info["observation"]["image"]  # type: ignore[index]
                page_screenshot_img = Image.fromarray(page_screenshot_arr)
                user_content: list[dict[str, Any]] = [
                    {"type": "text", "text": user_message},
                    {"type": "text", "text": "IMAGES: (1) current page screenshot"},
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(page_screenshot_img)},
                    },
                ]
                prompt: Any = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
            else:
                prompt = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
        else:
            # HuggingFace and other text-only providers: plain string.
            prompt = f"{system_prompt}\n\n{user_message}"

        try:
            response = call_llm(self.lm_config, prompt)
        except Exception:
            return action  # any LLM error → keep original

        # Parse the response for a backtick-delimited action.
        pattern = r"```((.|\n)*?)```"
        match = re.search(pattern, response)
        if not match:
            return action  # no parseable output → keep original

        parsed = match.group(1).strip()

        if parsed.lower() == "keep":
            return action

        # Attempt to build a new Action from the parsed string.
        try:
            if self.action_set_tag in [
                "id_accessibility_tree",
                "id_accessibility_tree_with_captioner",
                "som",
            ]:
                new_action = create_id_based_action(parsed)
            elif self.action_set_tag == "playwright":
                new_action = create_playwright_action(parsed)
            else:
                return action  # unknown tag → keep original
            new_action["raw_prediction"] = response
            return new_action
        except ActionParsingError:
            return action  # unparseable replacement → keep original

    def reset(self, test_config_file: str) -> None:
        pass


def construct_agent(args: argparse.Namespace, captioning_fn=None) -> Agent:
    llm_config = lm_config.construct_llm_config(args)

    agent: Agent
    if args.agent_type == "teacher_forcing":
        agent = TeacherForcingAgent()
    elif args.agent_type == "prompt":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn
        )
    else:
        raise NotImplementedError(
            f"agent type {args.agent_type} not implemented"
        )
    return agent
