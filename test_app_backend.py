import unittest
import json
from app import app, ai_engine

class TestWarAppBackend(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index_route(self):
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"WORLD WAR CHAIN REACTION MODEL", res.data)

    def test_api_chat_leader_query(self):
        res = self.client.post('/api/chat', json={"message": "Tell me about Adolf Hitler"})
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertIn("HITLER", data['reply'])
        self.assertEqual(data['type'], 'leader')

    def test_api_chat_war_query(self):
        res = self.client.post('/api/chat', json={"message": "What happened in the Eastern Front war?"})
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertIn("EASTERN FRONT", data['reply'])
        self.assertEqual(data['type'], 'war')

    def test_api_chat_country_query(self):
        res = self.client.post('/api/chat', json={"message": "What are the stats for Sudan?"})
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertIn("SUDAN", data['reply'])
        self.assertEqual(data['type'], 'country')

    def test_api_chat_predict_query(self):
        res = self.client.post('/api/chat', json={"message": "What is the early warning risk prediction?"})
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertIn("EARLY WARNING", data['reply'])

    def test_api_leaders_endpoint(self):
        res = self.client.get('/api/leaders')
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertTrue(len(data) >= 8)

    def test_api_war_interactions_endpoint(self):
        res = self.client.get('/api/war-interactions')
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertTrue(len(data) >= 7)

    def test_api_predict_endpoint(self):
        res = self.client.post('/api/predict', json={"polity": -8, "gdp_growth": -0.10, "deaths": 500})
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertIn("probability", data)
        self.assertIn("risk_level", data)

if __name__ == '__main__':
    unittest.main()
