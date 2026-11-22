
import sys
import json
import time
import pandas as pd

from pathlib import Path

# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR / "phishresearch_app"

sys.path.insert(0, str(APP_DIR))

import retrieval_engine
from llm_engine import generate_answer


# =====================================================
# 2. EVALUATION DATASET
# =====================================================

DATASET_DIR = BASE_DIR / "evaluation_dataset"

QUESTION_FILES = [
    "factual_evaluation_questions.csv",
    "conceptual_evaluation_questions.csv",
    "multi_paper_comparison_questions.csv",
    "unanswerable_questions.csv"
]

question_frames = []

for filename in QUESTION_FILES:

    df = pd.read_csv(DATASET_DIR / filename)

    df["Question_Set"] = Path(filename).stem

    question_frames.append(df)

questions = pd.concat(
    question_frames,
    ignore_index=True
)

questions["Question_ID"] = (
    questions["Question_ID"].astype(str)
)

if questions["Question_ID"].duplicated().any():
    raise ValueError("Duplicate question IDs detected.")

print("Total evaluation questions:", len(questions))


# =====================================================
# 3. EXPERIMENT CONFIGURATIONS
# =====================================================

CONFIGURATIONS = [
    "No Retrieval",
    "BM25",
    "SBERT",
    "Hybrid RRF + Cross-Encoder"
]

RETRIEVAL_METHODS = {
    "BM25": "BM25",
    "SBERT": "SBERT",
    "Hybrid RRF + Cross-Encoder":
        "Hybrid RRF + Cross-Encoder"
}

TOP_K = 5
MAX_NEW_TOKENS = 450

# First run: test only Q001 across all four configurations.
# After successful testing, change this to None for all 80.
MAX_QUESTIONS = None


# =====================================================
# 4. OUTPUT DIRECTORY
# =====================================================

OUTPUT_DIR = BASE_DIR / "llm_evaluation_results_local"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / "qwen_four_config_results.csv"

TEMP_FILE = OUTPUT_DIR / "results_temp.csv"


# =====================================================
# 5. LOAD PREVIOUS RESULTS FOR RESUMING
# =====================================================

if OUTPUT_FILE.exists():

    previous = pd.read_csv(OUTPUT_FILE)

    all_results = previous.to_dict("records")

    completed = set(
        zip(
            previous.loc[
                previous["Status"] == "SUCCESS",
                "Question_ID"
            ].astype(str),
            previous.loc[
                previous["Status"] == "SUCCESS",
                "Configuration"
            ]
        )
    )

    print("Previous successful results:", len(completed))

else:

    all_results = []
    completed = set()


def save_results():

    result_df = pd.DataFrame(all_results)

    result_df.to_csv(
        TEMP_FILE,
        index=False
    )

    TEMP_FILE.replace(OUTPUT_FILE)


# =====================================================
# 6. LOAD EXISTING RETRIEVAL ENGINE
# =====================================================

print("\nLoading retrieval engine...")

chunks, embeddings, bm25, embedding_model, cross_encoder = (
    retrieval_engine.load_engine()
)

print("Retrieval engine loaded successfully.")


# =====================================================
# 7. SELECT QUESTIONS
# =====================================================

if MAX_QUESTIONS is None:

    selected_questions = questions

else:

    selected_questions = questions.head(
        MAX_QUESTIONS
    )

total_expected = len(selected_questions) * len(CONFIGURATIONS)

print("\nQuestions selected:", len(selected_questions))
print("Configurations:", len(CONFIGURATIONS))
print("Expected answers:", total_expected)


# =====================================================
# 8. RUN THE EXPERIMENT
# =====================================================

for _, question_row in selected_questions.iterrows():

    question_id = str(question_row["Question_ID"])
    question = str(question_row["Question"])

    print("\n" + "=" * 70)
    print("QUESTION:", question_id)
    print(question)
    print("=" * 70)

    for configuration in CONFIGURATIONS:

        result_key = (question_id, configuration)

        if result_key in completed:

            print("\nAlready completed:", configuration)
            continue

        print("\nRunning:", configuration)

        total_start = time.time()

        results = pd.DataFrame()
        retrieval_time = 0.0
        generation_time = 0.0

        try:

            # -----------------------------------------
            # RETRIEVAL
            # -----------------------------------------

            if configuration == "No Retrieval":

                use_retrieval = False

            else:

                use_retrieval = True

                retrieval_start = time.time()

                results = retrieval_engine.retrieve(
                    query=question,
                    method=RETRIEVAL_METHODS[configuration],
                    top_k=TOP_K,
                    chunks=chunks,
                    bm25=bm25,
                    embeddings=embeddings,
                    embedding_model=embedding_model,
                    cross_encoder=cross_encoder
                )

                retrieval_time = (
                    time.time() - retrieval_start
                )

            # -----------------------------------------
            # QWEN ANSWER GENERATION
            # -----------------------------------------

            generation_start = time.time()

            answer = generate_answer(
                question=question,
                evidence_df=results,
                use_retrieval=use_retrieval,
                max_new_tokens=MAX_NEW_TOKENS
            )

            generation_time = (
                time.time() - generation_start
            )

            # -----------------------------------------
            # SAVE RETRIEVED EVIDENCE
            # -----------------------------------------

            evidence_records = []

            if not results.empty:

                for _, passage in results.iterrows():

                    evidence_records.append({
                        "Chunk_ID": passage.get("Chunk_ID"),
                        "Article_ID": passage.get("Article_ID"),
                        "Title": passage.get("Title"),
                        "Pages": passage.get("Pages"),
                        "Text": passage.get(
                            "Text",
                            passage.get("Text_retrieved", "")
                        )
                    })

            # -----------------------------------------
            # SAVE RESULT
            # -----------------------------------------

            record = {

                "Question_ID": question_id,

                "Question_Set": question_row["Question_Set"],

                "Question": question,

                "Category": question_row.get("Category"),

                "Answerable": question_row.get("Answerable"),

                "Reference_Answer": question_row.get(
                    "Reference_Answer"
                ),

                "Configuration": configuration,

                "Answer": answer,

                "Retrieved_Passage_Count": len(results),

                "Retrieved_Evidence": json.dumps(
                    evidence_records,
                    ensure_ascii=False,
                    default=str
                ),

                "Retrieval_Time_Seconds": round(
                    retrieval_time, 3
                ),

                "Generation_Time_Seconds": round(
                    generation_time, 3
                ),

                "Total_Time_Seconds": round(
                    time.time() - total_start, 3
                ),

                "Status": "SUCCESS",

                "Error": None
            }

            all_results.append(record)

            completed.add(result_key)

            save_results()

            print("Answer generated and saved.")
            print("Retrieved passages:", len(results))
            print("Generation time:", round(generation_time, 2), "seconds")

        except Exception as error:

            print("ERROR:", str(error))

            all_results.append({

                "Question_ID": question_id,

                "Question_Set": question_row["Question_Set"],

                "Question": question,

                "Category": question_row.get("Category"),

                "Answerable": question_row.get("Answerable"),

                "Reference_Answer": question_row.get(
                    "Reference_Answer"
                ),

                "Configuration": configuration,

                "Answer": None,

                "Retrieved_Passage_Count": len(results),

                "Retrieved_Evidence": "[]",

                "Retrieval_Time_Seconds": round(
                    retrieval_time, 3
                ),

                "Generation_Time_Seconds": round(
                    generation_time, 3
                ),

                "Total_Time_Seconds": round(
                    time.time() - total_start, 3
                ),

                "Status": "ERROR",

                "Error": str(error)
            })

            save_results()


# =====================================================
# 9. EXPERIMENT SUMMARY
# =====================================================

print("\n" + "=" * 70)
print("EXPERIMENT FINISHED")
print("=" * 70)

if OUTPUT_FILE.exists():

    final_df = pd.read_csv(OUTPUT_FILE)

    print("Saved results:", len(final_df))

    print("\nStatus summary:")
    print(final_df["Status"].value_counts())

    print("\nResults by configuration:")
    print(
        final_df.groupby(
            ["Configuration", "Status"]
        ).size()
    )

    print("\nOutput file:")
    print(OUTPUT_FILE)
