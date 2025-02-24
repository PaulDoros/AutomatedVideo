import re
from termcolor import colored
from typing import Tuple, Dict, List

class ContentQualityChecker:
    def __init__(self):
        self.min_words_per_sentence = 5
        self.max_words_per_sentence = 20
        self.ideal_duration = 30  # seconds for shorts
        self.words_per_second = 2.5  # average speaking rate

    def analyze_script(self, script: str) -> Tuple[bool, Dict]:
        """Analyze script quality and return metrics"""
        try:
            # Clean script
            script = re.sub(r'[\*\#\@\(\)\[\]]', '', script)
            sentences = [s.strip() for s in script.split('.') if s.strip()]
            
            # Calculate metrics
            metrics = {
                'total_sentences': len(sentences),
                'total_words': sum(len(s.split()) for s in sentences),
                'avg_words_per_sentence': sum(len(s.split()) for s in sentences) / len(sentences),
                'estimated_duration': sum(len(s.split()) for s in sentences) / self.words_per_second,
                'engagement_score': self._calculate_engagement(sentences),
                'issues': []
            }

            # Check for issues
            self._check_hook(script, metrics)  # Only check for hook
            self._check_engagement_elements(script, metrics)
            
            # Determine if content meets quality standards
            is_quality = (
                len(metrics['issues']) == 0 and
                metrics['engagement_score'] >= 7.0
            )

            return is_quality, metrics

        except Exception as e:
            print(colored(f"Error analyzing script: {str(e)}", "red"))
            return False, {'issues': ['Analysis failed']}

    def _check_hook(self, script: str, metrics: dict):
        """Check if script has a strong hook"""
        first_line = script.split('\n')[0].lower()
        if not any(h in first_line for h in ['?', 'want to', 'ever', 'how to', 'why', 'secret']):
            metrics['issues'].append('Missing strong hook at start')

    def _calculate_engagement(self, sentences: List[str]) -> float:
        """Calculate engagement score based on various factors"""
        score = 8.0  # Start with base score
        
        # Add points for engagement elements
        if any('?' in s for s in sentences): score += 1
        if any('!' in s for s in sentences): score += 1
        if any(emoji in ''.join(sentences) for emoji in ['💻', '🚀', '✨', '🔥']): score += 1
        
        return score

    def _check_sentence_length(self, sentences: List[str], metrics: Dict):
        """Check sentence length distribution"""
        for sentence in sentences:
            words = len(sentence.split())
            if words < self.min_words_per_sentence:
                metrics['issues'].append(f"Sentence too short: '{sentence}'")
            elif words > self.max_words_per_sentence:
                metrics['issues'].append(f"Sentence too long: '{sentence}'")

    def _check_duration(self, metrics: Dict):
        """Check if estimated duration is appropriate for shorts"""
        duration = metrics['estimated_duration']
        if duration < 15:
            metrics['issues'].append("Content too short for effective engagement")
        elif duration > 45:
            metrics['issues'].append("Content too long for shorts format")

    def _check_engagement_elements(self, script: str, metrics: Dict):
        """Check for necessary engagement elements"""
        script_lower = script.lower()
        
        # Check for call to action
        if not any(c in script_lower[-100:] for c in ['follow', 'subscribe', 'try this']):
            metrics['issues'].append("Missing call to action at end")
            
        # Check for questions/engagement
        if '?' not in script:
            metrics['issues'].append("No engaging questions found") 