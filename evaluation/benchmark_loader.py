# evaluation/benchmark_loader.py
"""Benchmark data loader for evaluation"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

import pandas as pd

@dataclass
class Question:
    """Single question data structure"""
    question_id: str
    question_group: str  # This is the category
    question_type: str  # "multiple_choice" or "open_ended"
    media_type: str  # "image" or "video" - important distinction
    question: str
    media_path: str  # Path to image or video
    ground_truth: str  # Correct answer for MC, reference answer for open questions
    options: Optional[List[str]] = None  # For multiple choice questions
    human_responses: Optional[List[str]] = None  # For LLM judge evaluation
    metadata: Optional[Dict] = None  # Contains annotators, data_source, media_source, etc.

class BenchmarkLoader:
    """Load and manage benchmark data with question groupings"""
    
    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        self.questions: List[Question] = []
        self.question_groups: Dict[str, List[Question]] = {}
        
    def load_from_json(self, json_file: str = "questions.json"):
        """Load questions from JSON file"""
        json_path = self.data_path / json_file
        
        if not json_path.exists():
            raise FileNotFoundError(f"Questions file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Load questions
        for item in data.get("questions", []):
            question = Question(
                question_id=item["question_id"],
                question_group=item["question_group"],  # This is the category
                question_type=item["question_type"],
                media_type=item["media_type"],  # "image" or "video"
                question=item["question_text"],
                media_path=item["media_path"],
                ground_truth=item["ground_truth"],
                options=item.get("options"),
                human_responses=item.get("human_responses"),
                metadata=item.get("metadata", {})
            )
            self.questions.append(question)
        
        # Organize by groups
        self._organize_by_groups()
        
        print(f"Loaded {len(self.questions)} questions from {json_path}")
        print(f"Question groups: {list(self.question_groups.keys())}")

    def load_from_csv(self, csv_file: str = "questions.csv"):
        """Load questions from CSV file"""
        csv_path = self.data_path / csv_file

        if not csv_path.exists():
            raise FileNotFoundError(f"Questions file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            # Parse JSON strings for list fields
            options = None
            if pd.notna(row.get("options")):
                options = json.loads(row["options"])

            human_responses = None
            if pd.notna(row.get("human_responses")):
                human_responses = json.loads(row["human_responses"])

            annotators = None
            if pd.notna(row.get("annotators")):
                annotators = json.loads(row["annotators"])

            # Reconstruct metadata from flattened columns
            metadata = {}
            if annotators:
                metadata["annotators"] = annotators
            if pd.notna(row.get("data_source")):
                metadata["data_source"] = row["data_source"]
            if pd.notna(row.get("difficulty")):
                metadata["difficulty"] = row["difficulty"]

            # Reconstruct media_source
            media_source = {}
            if pd.notna(row.get("video_id")):
                media_source["video_id"] = row["video_id"]
            if pd.notna(row.get("frame_number")):
                media_source["frame_number"] = int(row["frame_number"])
            if pd.notna(row.get("start_frame")):
                media_source["start_frame"] = int(row["start_frame"])
            if pd.notna(row.get("end_frame")):
                media_source["end_frame"] = int(row["end_frame"])
            if pd.notna(row.get("start_time")):
                media_source["start_time"] = row["start_time"]
            if pd.notna(row.get("end_time")):
                media_source["end_time"] = row["end_time"]
            if media_source:
                metadata["media_source"] = media_source

            question = Question(
                question_id=row["question_id"],
                question_group=row["question_group"],
                question_type=row["question_type"],
                media_type=row["media_type"],
                question=row["question_text"],
                media_path=row["media_path"] if pd.notna(row.get("media_path")) else None,
                ground_truth=row["ground_truth"],
                options=options,
                human_responses=human_responses,
                metadata=metadata if metadata else None,
            )
            self.questions.append(question)

        self._organize_by_groups()

        print(f"Loaded {len(self.questions)} questions from {csv_path}")
        print(f"Question groups: {list(self.question_groups.keys())}")

    def load(self, filename: str = "questions"):
        """Auto-detect and load from available format (CSV preferred, then JSON)"""
        for ext, loader in [(".csv", self.load_from_csv), (".json", self.load_from_json)]:
            path = self.data_path / f"{filename}{ext}"
            if path.exists():
                return loader(f"{filename}{ext}")
        raise FileNotFoundError(f"No questions file found matching: {self.data_path / filename}.*")

    def _organize_by_groups(self):
        """Organize questions by their groups"""
        for question in self.questions:
            group = question.question_group
            if group not in self.question_groups:
                self.question_groups[group] = []
            self.question_groups[group].append(question)
    
    def get_questions_by_group(self, group: str) -> List[Question]:
        """Get all questions for a specific group"""
        return self.question_groups.get(group, [])
    
    def get_all_questions(self) -> List[Question]:
        """Get all questions"""
        return self.questions
    
    def get_question_by_id(self, question_id: str) -> Optional[Question]:
        """Get a specific question by ID"""
        for question in self.questions:
            if question.question_id == question_id:
                return question
        return None
    
    def get_group_summary(self) -> Dict[str, Dict[str, int]]:
        """Get summary of question groups"""
        summary = {}
        for group, questions in self.question_groups.items():
            mc_count = sum(1 for q in questions if q.question_type == "multiple_choice")
            open_count = sum(1 for q in questions if q.question_type == "open_ended")
            image_count = sum(1 for q in questions if q.media_type == "image")
            video_count = sum(1 for q in questions if q.media_type == "video")
            summary[group] = {
                "total_questions": len(questions),
                "multiple_choice": mc_count,
                "open_ended": open_count,
                "image_questions": image_count,
                "video_questions": video_count
            }
        return summary
    
    def validate_questions(self) -> List[str]:
        """Validate loaded questions and return any issues"""
        issues = []
        
        for question in self.questions:
            # Check required fields
            if not question.question_id:
                issues.append(f"Question missing ID: {question}")
            
            if not question.question_group:
                issues.append(f"Question {question.question_id} missing group")
            
            if question.question_type not in ["multiple_choice", "open_ended"]:
                issues.append(f"Question {question.question_id} has invalid type: {question.question_type}")
            
            if question.media_type not in ["image", "video"]:
                issues.append(f"Question {question.question_id} has invalid media_type: {question.media_type}")
            
            # Check media file exists (skip if blind / no media)
            if question.media_path:
                media_path = self.data_path / question.media_path
                if not media_path.exists():
                    issues.append(f"Question {question.question_id} media file not found: {media_path}")
            
            # Check multiple choice questions have options
            if question.question_type == "multiple_choice" and not question.options:
                issues.append(f"Question {question.question_id} is multiple choice but has no options")
        
        return issues
    
