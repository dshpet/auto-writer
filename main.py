import os
import json
import logging
import re
import asyncio
from typing import Any, Dict, List, Callable
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# ----------------------------
# Environment and Logging Setup
# ----------------------------
load_dotenv()  # Loads environment variables (including OPENAI_API_KEY)

# Configure logging to output to both the console and a file.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("process.log", mode="a", encoding="utf-8")
    ]
)

# ----------------------------
# Configuration Constants
# ----------------------------
STATE_FILE = "state.json"
STORY_TXT_FILE = "iterative_story.txt"

LOWER_LIMIT = 15000      # Minimum character count for a "complete" story
UPPER_LIMIT = 20000      # Upper limit; if exceeded, further expansion is halted
IDEA_ITERATIONS = 3      # How many iterations in the idea phase before moving on
MAX_PHASE_ITERATIONS = 3  # Maximum iterations per phase

# Quality thresholds (0 to 100 scale) for each phase:
IDEA_THRESHOLD = 75.0
OUTLINE_THRESHOLD = 75.0
EXPANSION_THRESHOLD = 80.0
REFINEMENT_THRESHOLD = 95.0

# ----------------------------
# Dynamic Context Helpers
# ----------------------------
def get_context_char_limit(llm: ChatOpenAI, fraction: float = 1.0) -> int:
    """
    Returns an approximate character limit based on the LLM's max token setting.
    Uses a rough conversion of 1 token ≈ 4 characters.
    If the LLM does not expose max_tokens, defaults to 8192 tokens.
    """
    try:
        max_tokens = llm.max_tokens  # If available
    except AttributeError:
        max_tokens = 8192
    return int(max_tokens * 4 * fraction)

# ----------------------------
# Create LLM Instances
# ----------------------------
llm_temperatures = [0.7, 0.8, 0.9]
llms = [ChatOpenAI(model_name="gpt-4", temperature=temp) for temp in llm_temperatures]

# A robust LLM (with lower temperature) for refinement and meta evaluation.
robust_llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

# ----------------------------
# Helper Function: Async Invoke with Retry
# ----------------------------
async def async_invoke(chain: Any, params: Dict[str, Any], retries: int = 3, delay: float = 1.0) -> Any:
    """
    Asynchronously invoke a chain with the given parameters.
    Retries up to `retries` times with a delay between attempts.
    Uses asyncio.to_thread() to run the blocking call in a thread.
    """
    for attempt in range(retries):
        try:
            result = await asyncio.to_thread(chain.invoke, params)
            return result
        except Exception as e:
            logging.exception(f"Error invoking chain (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
    raise Exception("Max retries exceeded for chain invocation.")

# ----------------------------
# Helper Function to Build Chains
# ----------------------------
def build_chains(template: str, input_vars: List[str], llm_list: List[ChatOpenAI]) -> List[Any]:
    prompt = PromptTemplate(template=template, input_variables=input_vars)
    return [prompt | llm for llm in llm_list]

# ----------------------------
# Helper Function to Split Text into Chunks
# ----------------------------
def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into chunks of size `chunk_size` with an overlap.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# ----------------------------
# Define Prompt Templates
# ----------------------------
idea_template = (
    "Мета: {objective}\n\n"
    "Згенеруй креативну ідею для короткого кіберпанк оповідання, яке занурює читача у темний, технологічний світ майбутнього, "
    "де люди, корпорації та штучний інтелект борються за владу. Нехай ця ідея буде оригінальною, захоплюючою і має великий потенціал."
)

planning_template = (
    "Мета: {objective}\n\n"
    "Поточний запит/конспект:\n{current_text}\n\n"
    "На основі вищевикладеного згенеруй комплексний і детальний конспект для кіберпанк оповідання. "
    "Перерахуй основні сюжетні повороти, головні арки персонажів та ключові події у вигляді нумерованого списку."
)

expansion_template = (
    "Мета: {objective}\n\n"
    "Конспект (outline): {outline}\n\n"
    "Останній фрагмент історії:\n{chunk}\n\n"
    "Згенеруй наступний логічний фрагмент історії, продовжуючи попередній текст і дотримуючись конспекту."
)

refinement_template = (
    "Мета: {objective}\n\n"
    "Ідея:\n{idea}\n\n"
    "Конспект:\n{outline}\n\n"
    "Повна історія:\n{story}\n\n"
    "Оціни загальну цілісність, послідовність, креативність та стиль. Згенеруй покращену версію історії, "
    "зберігаючи всі деталі, але підвищуючи якість нарації. Поверни лише текст історії."
)

judge_template = (
    "Оцініть наступний текст за креативністю, послідовністю, структурою та залученістю. "
    "Використовуйте число від 0 до 100 і надайте короткий коментар. Текст:\n{candidate}"
)

council_meta_template = (
    "Нижче наведені відгуки кількох суддів про текст:\n"
    "{judges_feedback}\n"
    "Сирі оцінки: {raw_scores}\n"
    "Враховуючи ці відгуки, визначте фінальну оцінку для тексту '{candidate}' на шкалі від 0 до 100. "
    "Поверніть лише число."
)

# ----------------------------
# Build Chains
# ----------------------------
idea_chains = build_chains(idea_template, ["objective"], llms)
planning_chains = build_chains(planning_template, ["objective", "current_text"], llms)
expansion_chains = build_chains(expansion_template, ["objective", "outline", "chunk"], llms)
refinement_chain = build_chains(refinement_template, ["objective", "idea", "outline", "story"], [robust_llm])[0]
council_meta_chain = build_chains(council_meta_template, ["candidate", "judges_feedback", "raw_scores"], [robust_llm])[0]
judge_chains = build_chains(judge_template, ["candidate"], llms)

# ----------------------------
# State Persistence Functions
# ----------------------------
def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logging.exception("Error loading state")
    return {}

def save_state(state: Dict[str, Any]) -> None:
    serializable_state = {}
    for key, value in state.items():
        if isinstance(value, str):
            serializable_state[key] = value
        else:
            serializable_state[key] = value.content if hasattr(value, "content") else str(value)
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable_state, f, indent=2, ensure_ascii=False)
        logging.info("State saved successfully.")
    except Exception:
        logging.exception("Error saving state")

# ----------------------------
# Parsing Helper Functions
# ----------------------------
def parse_judge_response(result: Any) -> (float, str):
    text = result if isinstance(result, str) else result.content
    match = re.search(r"(\d+(\.\d+)?)", text)
    score = float(match.group(1)) if match else 0.0
    comment = text.replace(match.group(0), "") if match else text
    return score, comment.strip()

def parse_score(result: Any) -> float:
    text = result if isinstance(result, str) else result.content
    match = re.search(r"(\d+(\.\d+)?)", text)
    return float(match.group(1)) if match else 0.0

# ----------------------------
# Asynchronous Council Evaluation
# ----------------------------
async def async_evaluate_with_council(candidate: str) -> float:
    tasks = [async_invoke(judge, {"candidate": candidate}) for judge in judge_chains]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    scores = []
    feedbacks = []
    for res in results:
        if isinstance(res, Exception):
            logging.exception("Error in judge evaluation")
        else:
            score, feedback = parse_judge_response(res)
            scores.append(score)
            feedbacks.append(feedback)
    if not scores:
        return 0.0
    raw_scores_str = ", ".join(str(s) for s in scores)
    judges_feedback = "\n".join(feedbacks)
    logging.info(f"Candidate evaluation details:\nRaw scores: {raw_scores_str}\nFeedbacks: {judges_feedback}")
    try:
        meta_result = await async_invoke(council_meta_chain, {
            "candidate": candidate,
            "judges_feedback": judges_feedback,
            "raw_scores": raw_scores_str
        })
        final_score = parse_score(meta_result)
    except Exception:
        logging.exception("Error in council meta evaluation")
        final_score = sum(scores) / len(scores)
    return final_score

# ----------------------------
# Generic Asynchronous Phase Processing Function
# ----------------------------
async def async_process_phase(phase_name: str,
                              generate_fn: Callable[[Dict[str, Any]], Any],
                              threshold: float,
                              state: Dict[str, Any]) -> Dict[str, Any]:
    best_candidate = None
    best_score = 0.0
    iterations = 0
    while best_score < threshold and iterations < MAX_PHASE_ITERATIONS:
        candidates = await generate_fn(state)
        # Log all generated candidates (truncated for readability)
        for cand in candidates:
            logging.info(f"[{phase_name.upper()}] Generated candidate (first 200 chars): {cand[:200]}...")
        eval_tasks = [async_evaluate_with_council(candidate) for candidate in candidates]
        scores = await asyncio.gather(*eval_tasks, return_exceptions=True)
        for candidate, score in zip(candidates, scores):
            if isinstance(score, Exception):
                continue
            logging.info(f"[{phase_name.upper()}] Candidate score: {score} for candidate (first 200 chars): {candidate[:200]}...")
            if score > best_score:
                best_candidate = candidate
                best_score = score
        iterations += 1
        logging.info(f"{phase_name.capitalize()} phase iteration {iterations}, best score: {best_score}")
    if best_candidate:
        state[phase_name] = best_candidate
    return state

# ----------------------------
# Asynchronous Generation Functions for Each Phase
# ----------------------------
async def generate_candidates_idea(state: Dict[str, Any]) -> List[str]:
    tasks = [async_invoke(chain, {"objective": state["objective"]}) for chain in idea_chains]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    candidates = []
    for res in results:
        if isinstance(res, Exception):
            logging.exception("Error generating idea candidate")
        else:
            candidate = res if isinstance(res, str) else res.content.strip()
            candidates.append(candidate)
    return candidates

async def generate_candidates_outline(state: Dict[str, Any]) -> List[str]:
    current_outline = state.get("outline", "")
    tasks = [async_invoke(chain, {"objective": state["objective"], "current_text": current_outline})
             for chain in planning_chains]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    candidates = []
    for res in results:
        if isinstance(res, Exception):
            logging.exception("Error generating outline candidate")
        else:
            candidate = res if isinstance(res, str) else res.content.strip()
            candidates.append(candidate)
    return candidates

async def generate_candidates_expansion(state: Dict[str, Any]) -> List[str]:
    outline = state.get("outline", "")
    full_story = state.get("story", "")
    # Dynamically determine the chunk size for expansion using the first generation LLM.
    expansion_chunk_size = get_context_char_limit(llms[0], 0.3)
    chunk = full_story[-expansion_chunk_size:] if len(full_story) > expansion_chunk_size else full_story
    tasks = [async_invoke(chain, {"objective": state["objective"], "outline": outline, "chunk": chunk})
             for chain in expansion_chains]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    candidates = []
    for res in results:
        if isinstance(res, Exception):
            logging.exception("Error generating expansion candidate")
        else:
            candidate = res if isinstance(res, str) else res.content.strip()
            candidates.append(candidate)
    return candidates

async def generate_candidates_refinement(state: Dict[str, Any]) -> List[str]:
    story_text = state.get("story", "")
    # Dynamically get the robust model's context window in characters.
    robust_context_chars = get_context_char_limit(robust_llm, 1.0)
    # For refinement, leave room for idea and outline; use 80% of context for story.
    refine_chunk_size = get_context_char_limit(robust_llm, 0.8)
    refine_chunk_overlap = get_context_char_limit(robust_llm, 0.1)
    if len(story_text) > robust_context_chars:
        logging.info("Story is too long for a single refinement call; splitting into chunks...")
        chunks = split_text_into_chunks(story_text, refine_chunk_size, refine_chunk_overlap)
        refined_chunks = []
        for idx, chunk in enumerate(chunks):
            try:
                logging.info(f"Refining chunk {idx+1}/{len(chunks)} (first 200 chars): {chunk[:200]}...")
                res = await async_invoke(refinement_chain, {
                    "objective": state["objective"],
                    "idea": state.get("idea", ""),
                    "outline": state.get("outline", ""),
                    "story": chunk
                })
                refined_chunk = res if isinstance(res, str) else res.content.strip()
                refined_chunks.append(refined_chunk)
            except Exception:
                logging.exception(f"Error refining chunk {idx+1}")
        candidate = "\n".join(refined_chunks)
        return [candidate]
    else:
        try:
            res = await async_invoke(refinement_chain, {
                "objective": state["objective"],
                "idea": state.get("idea", ""),
                "outline": state.get("outline", ""),
                "story": story_text
            })
            candidate = res if isinstance(res, str) else res.content.strip()
            return [candidate]
        except Exception:
            logging.exception("Error generating refinement candidate")
            return []

# ----------------------------
# Asynchronous Phase Handlers
# ----------------------------
async def process_idea_phase(state: Dict[str, Any]) -> Dict[str, Any]:
    state = await async_process_phase("idea", generate_candidates_idea, IDEA_THRESHOLD, state)
    state["idea_iter"] = state.get("idea_iter", 0) + 1
    if state["idea_iter"] >= IDEA_ITERATIONS:
        logging.info("Idea phase complete. Moving to outline phase.")
        state["phase"] = "outline"
        state["outline"] = state.get("idea", "")
    return state

async def process_outline_phase(state: Dict[str, Any]) -> Dict[str, Any]:
    state = await async_process_phase("outline", generate_candidates_outline, OUTLINE_THRESHOLD, state)
    logging.info("Outline phase complete. Moving to expansion phase.")
    state["phase"] = "expansion"
    return state

async def process_expansion_phase(state: Dict[str, Any]) -> Dict[str, Any]:
    state = await async_process_phase("expansion", generate_candidates_expansion, EXPANSION_THRESHOLD, state)
    candidate = state.get("expansion", "")
    state["story"] = state.get("story", "") + "\n" + candidate
    logging.info(f"New chunk appended to story. Total length: {len(state.get('story', ''))} characters.")
    if len(state.get("story", "")) >= LOWER_LIMIT:
        state["phase"] = "refinement"
    return state

async def process_refinement_phase(state: Dict[str, Any]) -> Dict[str, Any]:
    state = await async_process_phase("refinement", generate_candidates_refinement, REFINEMENT_THRESHOLD, state)
    logging.info("Refinement phase complete. Story is finalized.")
    state["phase"] = "complete"
    return state

# Map phase names to their async handlers.
PHASE_HANDLERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "idea": process_idea_phase,
    "outline": process_outline_phase,
    "expansion": process_expansion_phase,
    "refinement": process_refinement_phase,
}

# ----------------------------
# Asynchronous Main Function
# ----------------------------
async def main() -> None:
    state: Dict[str, Any] = load_state()
    state["phase"] = state.get("phase", "idea")
    state["iteration"] = state.get("iteration", 0)
    state["idea"] = state.get("idea", "")
    state["idea_iter"] = state.get("idea_iter", 0)
    state["objective"] = state.get("objective", (
        "Напиши коротке кібепанк оповідання, яке занурює читача у темний, технологічний світ майбутнього, де люди, корпорації та штучний інтелект "
        "борються за владу. Нехай система сама згенерує конспект, основні сюжетні лінії, розвиток персонажів та всі деталі, "
        "необхідні для створення повноцінного твору."
    ))

    try:
        while True:
            if state.get("phase") == "complete":
                logging.info("Story creation process is complete.")
                break

            state["iteration"] += 1
            logging.info(f"=== Iteration {state['iteration']} | Phase: {state['phase']} ===")
            phase = state.get("phase")
            if phase in PHASE_HANDLERS:
                state = await PHASE_HANDLERS[phase](state)
            else:
                logging.error(f"Unknown phase: {phase}")
                break

            save_state(state)
            if "story" in state:
                try:
                    with open(STORY_TXT_FILE, "w", encoding="utf-8") as f:
                        f.write(state["story"])
                    logging.info(f"Story saved to {STORY_TXT_FILE}.")
                except Exception:
                    logging.exception("Error saving story to file")
    except KeyboardInterrupt:
        logging.info("Execution interrupted by user. Saving state and exiting gracefully.")
        save_state(state)

if __name__ == '__main__':
    asyncio.run(main())