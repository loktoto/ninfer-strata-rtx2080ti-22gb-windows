import unittest

from tools.privacy_scan import find_issues


class PrivacyScanTests(unittest.TestCase):
    def test_reserved_example_domain_is_allowed(self):
        self.assertFalse(find_issues("test@example.invalid"))

    def test_real_email_is_flagged_without_logging_value(self):
        self.assertTrue(find_issues("person@" + "private-mail.tld"))

    def test_windows_user_directory_is_flagged(self):
        self.assertTrue(find_issues("C:" + "\\Users\\Someone\\Documents"))

    def test_private_key_marker_is_flagged(self):
        self.assertTrue(find_issues("-----BEGIN " + "PRIVATE KEY-----"))

    def test_normal_source_is_allowed(self):
        self.assertFalse(find_issues("model_path = 'D:/AI/LLM/models/sage/model.gguf'"))


if __name__ == "__main__":
    unittest.main()
