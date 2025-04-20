import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os
import argparse
import logging

def processDaquarDataset(
        dataset_folder="data",
        raw_qa_file="raw/qa_pairs.txt",
        train_file="data_train.csv",
        eval_file="data_eval.csv",
        answer_space_file="answer_space.txt",
        test_size=0.2,
        random_seed=42
    ):
    
    logging.basicConfig(level=logging.INFO)
    
    # Regular expression to extract image ID from questions
    image_pattern = re.compile("( (in |on |of )?(the |this )?(image\d*) \?)")
    
    # Load raw question-answer pairs
    qa_path = os.path.join(dataset_folder, raw_qa_file)
    with open(qa_path) as f:
        qa_data = [x.replace("\n", "") for x in f.readlines()]
    logging.info(f"Loaded {len(qa_data)//2} question-answer pairs from {qa_path}")
    
    # Create dataframe for processed data
    df = pd.DataFrame({"question": [], "answer": [], "image_id": []})
    
    # Process raw QA pairs
    logging.info("Processing raw QnA pairs...")
    for i in range(0, len(qa_data), 2):
        try:
            img_id = image_pattern.findall(qa_data[i])[0][3]
            question = qa_data[i].replace(image_pattern.findall(qa_data[i])[0][0], "")
            record = {
                "question": question,
                "answer": qa_data[i+1],
                "image_id": img_id,
            }
            df = df.append(record, ignore_index=True)
        except IndexError:
            logging.warning(f"Could not parse question: {qa_data[i]}")
    
    logging.info(f"Processed {len(df)} question-answer pairs")
    
    # Create answer space vocabulary
    logging.info("Creating space of all possible answers")
    answer_space = []
    for ans in df.answer.to_list():
        answer_space = answer_space + [ans] if "," not in ans else answer_space + ans.replace(" ", "").split(",") 
    answer_space = list(set(answer_space))
    answer_space.sort()
    
    # Save answer space vocabulary
    answer_space_path = os.path.join(dataset_folder, answer_space_file)
    with open(answer_space_path, "w") as f:
        f.writelines("\n".join(answer_space))
    logging.info(f"Created answer space with {len(answer_space)} unique answers in {answer_space_path}")
    
    # Split into train and eval sets
    logging.info(f"Splitting into train & eval sets (test_size={test_size})")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed)
    
    # Save processed datasets
    train_path = os.path.join(dataset_folder, train_file)
    eval_path = os.path.join(dataset_folder, eval_file)
    train_df.to_csv(train_path, index=None)
    test_df.to_csv(eval_path, index=None)
    logging.info(f"Saved {len(train_df)} training examples to {train_path}")
    logging.info(f"Saved {len(test_df)} evaluation examples to {eval_path}")
    
    return {
        "train_size": len(train_df),
        "eval_size": len(test_df),
        "answer_space_size": len(answer_space)
    }
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DAQUAR dataset for VQA")
    parser.add_argument("--dataset_folder", type=str, default="data", 
                        help="Folder containing the dataset")
    parser.add_argument("--raw_qa_file", type=str, default="raw/qa_pairs.txt", 
                        help="Path to raw QA pairs relative to dataset folder")
    parser.add_argument("--train_file", type=str, default="data_train.csv", 
                        help="Output file for training data")
    parser.add_argument("--eval_file", type=str, default="data_eval.csv", 
                        help="Output file for evaluation data")
    parser.add_argument("--answer_space_file", type=str, default="answer_space.txt", 
                        help="Output file for answer vocabulary")
    parser.add_argument("--test_size", type=float, default=0.2, 
                        help="Proportion of data to use for evaluation")
    parser.add_argument("--random_seed", type=int, default=42, 
                        help="Random seed for train/test split")
    
    args = parser.parse_args()
    
    processDaquarDataset(
        dataset_folder=args.dataset_folder,
        raw_qa_file=args.raw_qa_file,
        train_file=args.train_file,
        eval_file=args.eval_file,
        answer_space_file=args.answer_space_file,
        test_size=args.test_size,
        random_seed=args.random_seed
    )