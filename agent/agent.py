import argparse
import dataclasses
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

        # Rolling history of stop-eval decisions for use as in-context examples.
        self._stop_eval_history: list[str] = []

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

        # Build a summary of the last ≤3 stop-eval decisions for the prompt.
        recent_history = self._stop_eval_history[-3:]
        if recent_history:
            stop_eval_history_str = (
                "PREVIOUS STOP-EVAL DECISIONS (most recent last):\n"
                + "\n".join(recent_history)
                + "\n\n"
            )
        else:
            stop_eval_history_str = ""

        if is_stop:
            system_prompt = (
                "You are a critical evaluator for a web automation agent. "
                "The agent has decided to STOP and submit a final answer.\n"
                "Your job: verify whether this stop decision is truly correct given the current page state.\n\n"
                "Reason step-by-step through the following:\n"
                "1. What exactly does the objective require? "
                "(a specific piece of information, a completed action, a navigation target, etc.)\n"
                "2. Is the required information or confirmation of completion clearly visible "
                "in the current observation? Like for prices, names, counts, colors, statuses, etc.\n"
                "3. Did the agent reach the correct page to stop? "
                "The STOP criteria is STRICT:\n"
                "   - MUST be on the exact item/post/listing page if the task requires it "
                "(e.g. EVEN IF you find the item, you HAVE TO CLICK INTO that exact item page/URL to be considered completed)"
                "NOT a search results page, category page, or overview page.\n"
                "   - ONLY IF it is impossible to reach the exact item page (e.g. no direct link exists, or summary is required), "
                "stop at the best available page that contains the answer.\n\n"
                "Output format — reply with your reasoning, then end with EXACTLY ONE of with triple backticks:\n"
                "- ```keep``` if stopping is correct and the answer is accurate.\n"
                "- ```<next_action>``` (e.g. ```click [42]```, ```scroll [down]```) if the task is NOT yet complete. "
                "Choose the most logical next action based on the current observation."
            )
            user_message = (
                f"OBJECTIVE: {intent}\n\n"
                f"ACTION HISTORY (all steps taken so far):\n{action_history_str}\n\n"
                f"{stop_eval_history_str}"
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
                "(a specific piece of information, a completed action, a navigation target, etc.)\n"
                "2. Is the required information or confirmation of completion clearly visible "
                "in the current observation? Like for prices, names, counts, colors, statuses, etc.\n"
                "3. Has the agent reached the correct page to stop? "
                "The STOP criteria is STRICT:\n"
                "   - MUST be on the exact item/post/listing page if the task requires it "
                "(e.g. the product detail page, the specific post page) — NOT a search results page, "
                "category page, or overview page. Stopping anywhere else will FAIL the task.\n"
                "   - ONLY IF it is impossible to reach the exact item page (e.g. no direct link exists, or summary is required), "
                "stop at the best available page that contains the answer and state it clearly.\n"
                "4. Would the agent's proposed next action add new information or progress, "
                "or is it redundant given what is already visible?\n\n"
                "Output format — reply with your reasoning, then end with EXACTLY ONE of with triple backticks:\n"
                "- ```keep``` if the task is not yet complete and the agent should continue.\n"
                "- ```stop [answer]``` if the task IS already complete. "
                "The answer must be the exact value the objective asked for "
                '(e.g., a price like "$12.99", a name, a count, "N/A", '
                'or "done" for action tasks).'
            )
            user_message = (
                f"OBJECTIVE: {intent}\n\n"
                f"ACTION HISTORY (all steps taken so far):\n{action_history_str}\n\n"
                f"{stop_eval_history_str}"
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

        # Reasoning models (e.g. gpt-5-mini) consume hidden thinking tokens within
        # max_completion_tokens. 384 (the default) leaves nothing for visible output.
        # Use a higher limit specifically for the evaluator call.
        eval_lm_config = dataclasses.replace(
            self.lm_config,
            gen_config={**self.lm_config.gen_config, "max_tokens": 2048},
        )
        try:
            response = call_llm(eval_lm_config, prompt)
        except Exception:
            return action  # any LLM error → keep original

        case_label = "STOP" if is_stop else "non-STOP"
        original_action_str = action.get("answer") if is_stop else action.get("raw_prediction", str(action))
        print(f"[stop_eval | {case_label}] Agent action: {original_action_str}", flush=True)
        print(f"[stop_eval | {case_label}] Evaluator response: {response}", flush=True)

        # Parse the response for an action string.
        # Priority 1: backtick-delimited  ```action```
        # Priority 2: last non-empty line (models routinely omit backtick
        #             formatting and just write "keep" or "click [16]" etc.)
        pattern = r"```((.|\n)*?)```"
        match = re.search(pattern, response)
        if match:
            parsed = match.group(1).strip()
        else:
            last_line = next(
                (l.strip() for l in reversed(response.splitlines()) if l.strip()),
                "",
            )
            if last_line:
                parsed = last_line
            else:
                print("[stop_eval] no parseable action found — keeping original", flush=True)
                return action

        if parsed.lower() == "keep":
            print("[stop_eval] decision: keep original action", flush=True)
            self._stop_eval_history.append(
                f"[{case_label}] Agent: {original_action_str!r}\n"
                f"Evaluator reasoning: {response}\n"
                f"Decision: keep"
            )
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
                print("[stop_eval] unknown action_set_tag — keeping original", flush=True)
                return action  # unknown tag → keep original
            print(f"[stop_eval] decision: override action → {parsed!r}", flush=True)
            self._stop_eval_history.append(
                f"[{case_label}] Agent: {original_action_str!r}\n"
                f"Evaluator reasoning: {response}\n"
                f"Decision: override with {parsed!r}"
            )
            new_action["raw_prediction"] = response
            return new_action
        except ActionParsingError:
            print(f"[stop_eval] could not parse replacement {parsed!r} — keeping original", flush=True)
            return action  # unparseable replacement → keep original

    def reset(self, test_config_file: str) -> None:
        self._stop_eval_history = []


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
