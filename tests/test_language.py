"""D3 lexical positives, hard negatives, and manual-annotation boundaries."""

import json
from pathlib import Path
import tempfile
import unittest

from src.speech.events import SpeechEvent
from src.speech.language import DEFAULT_LEXICON, LanguageCueExtractor, enrich_event, extract_cues


class LanguageCueTests(unittest.TestCase):
    def test_bilingual_aliases_map_to_detector_categories(self):
        examples = {
            "大家看看这个水杯": ["cup"],
            "这是马克杯和水瓶": ["cup", "bottle"],
            "Here is the APPLE": ["apple"],
            "a water\tcup, a MUG, and two bottles": ["cup", "bottle"],
            "手机和书本放在橙子旁": ["cell phone", "book", "orange"],
        }
        for text, classes in examples.items():
            with self.subTest(text=text):
                self.assertEqual(extract_cues(text).mentioned_classes, classes)

    def test_class_mentions_alone_are_not_presentation(self):
        for text in ("apple", "桌上有杯子和苹果", "This is a bottle", "I drink from a mug"):
            with self.subTest(text=text):
                self.assertTrue(extract_cues(text).mentioned_classes)
                self.assertNotIn("PRESENTATION_LANGUAGE", extract_cues(text).cue_types)

    def test_regional_cup_and_bottle_aliases_are_category_words_only(self):
        for text, expected in (("保温杯", "cup"), ("保溫杯", "cup"),
                               ("宝特瓶", "bottle"), ("寶特瓶", "bottle"),
                               ("塑料瓶", "bottle"), ("塑膠瓶", "bottle")):
            with self.subTest(text=text):
                cues = extract_cues(text)
                self.assertEqual(cues.mentioned_classes, [expected])
                self.assertNotIn("PRESENTATION_LANGUAGE", cues.cue_types)

    def test_traditional_aliases_and_presentation_preserve_original_text(self):
        for text, classes in (
            ("請大家看看這個蘋果", ["apple"]),
            ("給大家看這個馬克杯", ["cup"]),
            ("來看看保溫瓶和礦泉水瓶", ["bottle"]),
            ("看這裡的手機和這本書", ["cell phone", "book"]),
        ):
            with self.subTest(text=text):
                original = SpeechEvent(100, 500, 750, text)
                enriched = enrich_event(original)
                self.assertEqual(enriched.mentioned_classes, classes)
                self.assertIn("PRESENTATION_LANGUAGE", enriched.cue_types)
                self.assertEqual(enriched.text, text)
                self.assertEqual((enriched.start_ms, enriched.end_ms, enriched.available_at_ms), (100, 500, 750))

    def test_traditional_negative_and_topic_expressions_remain_conservative(self):
        for text in ("先不展示這個蘋果", "別看這個馬克杯", "不是這個蘋果，給大家看杯子", "沒有給大家看這個水瓶"):
            with self.subTest(text=text):
                cues = extract_cues(text)
                self.assertIn("NEGATED_LANGUAGE", cues.cue_types)
                self.assertNotIn("PRESENTATION_LANGUAGE", cues.cue_types)
        for text in ("換下一個", "換一個物品", "把蘋果移到旁邊", "水杯放到旁邊"):
            with self.subTest(text=text):
                self.assertIn("TOPIC_CHANGE", extract_cues(text).cue_types)

    def test_traditional_compounds_are_excluded_but_plain_phone_matches(self):
        for text in ("蘋果汁", "蘋果手機", "蘋果電腦", "蘋果醬"):
            with self.subTest(text=text):
                self.assertEqual(extract_cues(text).mentioned_classes, [])
        self.assertEqual(extract_cues("這是我的手機").mentioned_classes, ["cell phone"])
        self.assertEqual(extract_cues("蘋果汁旁邊有蘋果和手機").mentioned_classes, ["apple", "cell phone"])

    def test_presentation_expressions_are_separate_from_class_names(self):
        for text in ("给大家看一下这个苹果", "看看这个杯子", "看这里", "let me show you this bottle", "Look at this mug", "HERE'S the apple"):
            with self.subTest(text=text):
                self.assertIn("PRESENTATION_LANGUAGE", extract_cues(text).cue_types)
        self.assertEqual(extract_cues("看这里").mentioned_classes, [])

    def test_negation_suppresses_positive_expression_across_clause(self):
        for text in (
            "先不展示苹果", "不要看这个杯子", "不是这个苹果，给大家看杯子",
            "这次没有展示水瓶", "Do not show the apple", "Don't look at this mug",
            "I am not showing the bottle", "Let me show this apple, no, not this one",
            "I won't show the book",
        ):
            with self.subTest(text=text):
                cues = extract_cues(text)
                self.assertIn("NEGATED_LANGUAGE", cues.cue_types)
                self.assertNotIn("PRESENTATION_LANGUAGE", cues.cue_types)

    def test_recorded_closing_phrase_is_ending_cue_and_preserves_event(self):
        text = "今天就先給大家展示到這裡"
        original = SpeechEvent(100, 500, 750, text, [], ["PRESENTATION_LANGUAGE"])
        enriched = enrich_event(original)
        self.assertEqual(enriched.cue_types, ["ENDING_LANGUAGE"])
        self.assertEqual(enriched.text, text)
        self.assertEqual((enriched.start_ms, enriched.end_ms, enriched.available_at_ms, enriched.is_final),
                         (100, 500, 750, True))
        self.assertEqual(original.cue_types, ["PRESENTATION_LANGUAGE"])
        self.assertNotIn("state", enriched.to_dict())

    def test_simplified_traditional_and_english_closings_suppress_presentation(self):
        for text in (
            "今天先给大家展示到这里。", "本次演示到這裡為止。", "這個蘋果的展示結束了。",
            "展示完毕。", "我们现在结束展示。", "現在停止演示。",
            "That's all for today.", "That's it for this demonstration.",
            "The presentation is now over.", "Our demo has finished.",
            "I am done showing this cup.", "We're finished presenting the apple.",
            "We will stop showing this bottle.", "Let's end this demonstration.",
        ):
            with self.subTest(text=text):
                cues = extract_cues(text)
                self.assertIn("ENDING_LANGUAGE", cues.cue_types)
                self.assertNotIn("PRESENTATION_LANGUAGE", cues.cue_types)

    def test_negated_closing_is_neither_positive_ending_nor_presentation(self):
        for text in (
            "不要结束展示", "還沒展示完這個蘋果", "还没展示完", "展示還沒結束",
            "不要結束展示，給大家看這個杯子", "尚未完成展示",
            "I am not done showing this cup.", "We haven't finished showing the apple.",
            "Don't end this demonstration.", "The demonstration is not over.",
        ):
            with self.subTest(text=text):
                cues = extract_cues(text)
                self.assertIn("NEGATED_LANGUAGE", cues.cue_types)
                self.assertNotIn("ENDING_LANGUAGE", cues.cue_types)
                self.assertNotIn("PRESENTATION_LANGUAGE", cues.cue_types)

    def test_closing_with_new_positive_clause_is_conservatively_rejected(self):
        for text in ("今天展示到这里，给大家看看下一个杯子。",
                     "Let me show this apple. That's all for today."):
            with self.subTest(text=text):
                cues = extract_cues(text)
                self.assertIn("ENDING_LANGUAGE", cues.cue_types)
                self.assertNotIn("PRESENTATION_LANGUAGE", cues.cue_types)

    def test_ordinary_presentation_and_mentions_do_not_become_ending(self):
        for text in ("給大家看這個蘋果", "大家看看這個杯子", "Let me show you this bottle."):
            with self.subTest(text=text):
                self.assertIn("PRESENTATION_LANGUAGE", extract_cues(text).cue_types)
                self.assertNotIn("ENDING_LANGUAGE", extract_cues(text).cue_types)
        for text in ("这个苹果", "展示柜在这里", "结束展示按钮在这里", "如何结束展示", "我在解释怎么结束展示",
                     "The demonstration model is over there.", "When will the demonstration be over?"):
            with self.subTest(text=text):
                self.assertNotIn("ENDING_LANGUAGE", extract_cues(text).cue_types)

    def test_english_aliases_require_whole_words(self):
        self.assertEqual(extract_cues("pineapple cupboard bottled booking orangesicle").mentioned_classes, [])
        self.assertEqual(extract_cues("pineapple, then apple; cupboard, then cup.").mentioned_classes, ["apple", "cup"])

    def test_compounds_are_masked_without_hiding_later_independent_object(self):
        for text in ("苹果汁", "苹果醋", "苹果手机", "apple juice", "apple pie", "orange juice"):
            with self.subTest(text=text):
                self.assertEqual(extract_cues(text).mentioned_classes, [])
        self.assertEqual(extract_cues("苹果汁旁有苹果").mentioned_classes, ["apple"])
        self.assertEqual(extract_cues("apple juice and one apple").mentioned_classes, ["apple"])

    def test_mentions_are_unique_and_follow_text_order(self):
        self.assertEqual(extract_cues("瓶子、苹果、马克杯、水杯、bottle、apple").mentioned_classes,
                         ["bottle", "apple", "cup"])

    def test_topic_change_is_reported_as_lexical_cue(self):
        for text in ("把苹果移到旁边", "先放下杯子", "Let's move on", "put the bottle down"):
            with self.subTest(text=text):
                self.assertIn("TOPIC_CHANGE", extract_cues(text).cue_types)

    def test_enrichment_keeps_timing_and_replaces_fixture_annotations(self):
        original = SpeechEvent(100, 500, 750, "先不展示水杯", ["apple"], ["PRESENTATION_LANGUAGE"])
        enriched = enrich_event(original)
        self.assertEqual((enriched.start_ms, enriched.end_ms, enriched.available_at_ms, enriched.text, enriched.is_final),
                         (100, 500, 750, original.text, True))
        self.assertEqual(enriched.mentioned_classes, ["cup"])
        self.assertIn("NEGATED_LANGUAGE", enriched.cue_types)
        self.assertNotIn("PRESENTATION_LANGUAGE", enriched.cue_types)
        self.assertEqual(original.mentioned_classes, ["apple"])
        self.assertNotIn("target_id", enriched.to_dict())

    def test_custom_aliases_and_regex_are_configurable(self):
        config = json.loads(DEFAULT_LEXICON.read_text(encoding="utf-8"))
        config["classes"] = {"cup": ["tumbler"]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rules.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            extractor = LanguageCueExtractor(path)
            self.assertEqual(extractor.extract("look at this tumbler").mentioned_classes, ["cup"])
            self.assertEqual(extractor.extract("apple").mentioned_classes, [])
            config["presentation_patterns"] = ["("]
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid regex"):
                LanguageCueExtractor(path)

    def test_ending_patterns_are_optional_in_existing_version_one_lexicons(self):
        config = json.loads(DEFAULT_LEXICON.read_text(encoding="utf-8"))
        config.pop("ending_patterns")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy_rules.json"
            for ending in (None, []):
                if ending is not None:
                    config["ending_patterns"] = ending
                path.write_text(json.dumps(config), encoding="utf-8")
                extractor = LanguageCueExtractor(path)
                self.assertIn("PRESENTATION_LANGUAGE", extractor.extract("给大家看这个苹果").cue_types)
                self.assertNotIn("ENDING_LANGUAGE", extractor.extract("今天展示到这里").cue_types)
            config["ending_patterns"] = ["("]
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ending_patterns"):
                LanguageCueExtractor(path)

    def test_empty_text_is_valid_but_nontext_is_rejected(self):
        self.assertEqual(extract_cues("").to_dict(), {"mentioned_classes": [], "cue_types": []})
        for text in (None, 1, []):
            with self.subTest(text=text), self.assertRaises(ValueError):
                extract_cues(text)


if __name__ == "__main__":
    unittest.main()
