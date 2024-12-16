
    <pre>
      import threading
      import time
      from flask import Flask, jsonify
      import requests
      import sqlite3
      import numpy as np
      import tensorflow as tf
      from kivy.app import App
      from kivy.uix.screenmanager import ScreenManager, Screen
      from kivy.lang import Builder
      from kivy.clock import Clock
      from kivy.properties import StringProperty, NumericProperty
      from kivymd.app import MDApp
      from kivymd.uix.navigationdrawer import MDNavigationDrawer
      from kivymd.uix.list import OneLineListItem
      from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
      import matplotlib.pyplot as plt
      from matplotlib.figure import Figure
      import io
      from PIL import Image

      # Flask Backend
      app = Flask(__name__)
      COINGECKO_API_URL = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=true&price_change_percentage=24h,7d"
      crypto_data_cache = {}
      last_cache_update = 0
      CACHE_EXPIRY = 5

      def fetch_crypto_data():
          global crypto_data_cache, last_cache_update
          if time.time() - last_cache_update > CACHE_EXPIRY:
              try:
                  response = requests.get(COINGECKO_API_URL)
                  response.raise_for_status()
                  crypto_data_cache = response.json()
                  last_cache_update = time.time()
              except requests.exceptions.RequestException as e:
                  print(f"Error fetching data: {e}")
                  crypto_data_cache = {}
          return crypto_data_cache

      @app.route('/crypto_data')
      def get_crypto_data():
          data = fetch_crypto_data()
          return jsonify(data)

      @app.route('/predict/<string:symbol>')
      def predict_price(symbol):
          try:
              conn = sqlite3.connect('crypto_data.db')
              cursor = conn.cursor()
              cursor.execute("SELECT price FROM historical_data WHERE symbol = ? ORDER BY timestamp DESC LIMIT 100", (symbol,))
              historical_prices = cursor.fetchall()
              conn.close()
              if not historical_prices or len(historical_prices) < 50:
                  return jsonify({"prediction": "Not enough data"})
              prices = np.array([price[0] for price in historical_prices]).reshape(-1, 1)
              prices = (prices - np.mean(prices)) / np.std(prices)
              prices = prices[-50:].reshape(1, 50, 1)
              prediction = model.predict(prices)[0][0]
              prediction = prediction * np.std(prices) + np.mean(prices)
              return jsonify({"prediction": float(prediction)})
          except Exception as e:
              return jsonify({"error": str(e)})

      def run_flask_app():
          app.run(debug=False, use_reloader=False, port=5000)

      # SQLite Database
      def create_database():
          conn = sqlite3.connect('crypto_data.db')
          cursor = conn.cursor()
          cursor.execute('''
              CREATE TABLE IF NOT EXISTS historical_data (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  symbol TEXT,
                  price REAL,
                  timestamp INTEGER
              )
          ''')
          conn.commit()
          conn.close()

      def store_historical_data(data):
          conn = sqlite3.connect('crypto_data.db')
          cursor = conn.cursor()
          for item in data:
              symbol = item['symbol'].upper()
              price = item['current_price']
              timestamp = int(time.time())
              cursor.execute("INSERT INTO historical_data (symbol, price, timestamp) VALUES (?, ?, ?)", (symbol, price, timestamp))
          conn.commit()
          conn.close()

      def prune_historical_data():
          conn = sqlite3.connect('crypto_data.db')
          cursor = conn.cursor()
          cutoff_timestamp = int(time.time()) - (30 * 24 * 3600)
          cursor.execute("DELETE FROM historical_data WHERE timestamp < ?", (cutoff_timestamp,))
          conn.commit()
          conn.close()

      # AI Model (Pre-trained)
      def create_lstm_model():
          model = tf.keras.models.Sequential([
              tf.keras.layers.LSTM(50, activation='relu', input_shape=(50, 1)),
              tf.keras.layers.Dense(1)
          ])
          model.compile(optimizer='adam', loss='mse')
          return model

      model = create_lstm_model()
      # Load pre-trained weights (replace with actual weights)
      # model.load_weights('lstm_weights.h5')

      # Kivy App
      class CryptoApp(MDApp):
          def build(self):
              self.theme_cls.theme_style = "Dark"
              self.theme_cls.primary_palette = "BlueGray"
              self.screen_manager = ScreenManager()
              self.crypto_screen = CryptoScreen(name='crypto')
              self.trading_screen = TradingScreen(name='trading')
              self.prediction_screen = PredictionScreen(name='prediction')
              self.screen_manager.add_widget(self.crypto_screen)
              self.screen_manager.add_widget(self.trading_screen)
              self.screen_manager.add_widget(self.prediction_screen)
              return self.screen_manager

          def on_start(self):
              self.load_crypto_data()
              Clock.schedule_interval(self.load_crypto_data, 5)
              Clock.schedule_interval(self.update_graphs, 5)

          def load_crypto_data(self, dt=None):
              threading.Thread(target=self._load_crypto_data).start()

          def _load_crypto_data(self):
              try:
                  response = requests.get("http://127.0.0.1:5000/crypto_data")
                  response.raise_for_status()
                  data = response.json()
                  self.crypto_screen.update_crypto_list(data)
                  store_historical_data(data)
                  prune_historical_data()
              except requests.exceptions.RequestException as e:
                  print(f"Error loading data: {e}")

          def update_graphs(self, dt=None):
              self.crypto_screen.update_graphs()

      class CryptoScreen(Screen):
          crypto_list = StringProperty()
          def __init__(self, **kwargs):
              super().__init__(**kwargs)
              self.crypto_data = []
              self.graph_images = {}

          def update_crypto_list(self, data):
              self.crypto_data = data
              list_items = []
              for item in data:
                  symbol = item['symbol'].upper()
                  price = item['current_price']
                  change_24h = item['price_change_percentage_24h']
                  change_7d = item['price_change_percentage_7d_in_currency']
                  list_items.append(f"{symbol}: ${price:.2f} (24h: {change_24h:.2f}%, 7d: {change_7d:.2f}%)")
              self.crypto_list = "\n".join(list_items)

          def update_graphs(self):
              for item in self.crypto_data:
                  symbol = item['symbol'].upper()
                  sparkline = item['sparkline_in_7d']['price']
                  if sparkline:
                      self.graph_images[symbol] = self.create_graph(sparkline)
                      self.ids[f'graph_{symbol}'].source = self.graph_images[symbol]

          def create_graph(self, prices):
              fig = Figure(figsize=(4, 2), dpi=100)
              ax = fig.add_subplot(111)
              ax.plot(prices)
              ax.set_xticks([])
              ax.set_yticks([])
              ax.set_facecolor('#222222')
              fig.patch.set_facecolor('#222222')
              buf = io.BytesIO()
              fig.savefig(buf, format='png')
              buf.seek(0)
              image = Image.open(buf)
              image_path = f'graph_{time.time()}.png'
              image.save(image_path)
              return image_path

      class TradingScreen(Screen):
          def __init__(self, **kwargs):
              super().__init__(**kwargs)
              self.portfolio = {}
              self.balance = 10000
              self.transactions = []
              self.update_dashboard()

          def buy_crypto(self, symbol, amount):
              try:
                  response = requests.get("http://127.0.0.1:5000/crypto_data")
                  response.raise_for_status()
                  data = response.json()
                  for item in data:
                      if item['symbol'].upper() == symbol.upper():
                          price = item['current_price']
                          cost = price * amount
                          if self.balance >= cost:
                              self.balance -= cost
                              if symbol in self.portfolio:
                                  self.portfolio[symbol] += amount
                              else:
                                  self.portfolio[symbol] = amount
                              self.transactions.append(f"Bought {amount} {symbol} at ${price:.2f}")
                              self.update_dashboard()
                              return
                          else:
                              self.ids.trading_output.text = "Insufficient funds"
                              return
                  self.ids.trading_output.text = "Crypto not found"
              except requests.exceptions.RequestException as e:
                  self.ids.trading_output.text = f"Error: {e}"

          def sell_crypto(self, symbol, amount):
              try:
                  response = requests.get("http://127.0.0.1:5000/crypto_data")
                  response.raise_for_status()
                  data = response.json()
                  for item in data:
                      if item['symbol'].upper() == symbol.upper():
                          price = item['current_price']
                          if symbol in self.portfolio and self.portfolio[symbol] >= amount:
                              self.portfolio[symbol] -= amount
                              self.balance += price * amount
                              self.transactions.append(f"Sold {amount} {symbol} at ${price:.2f}")
                              self.update_dashboard()
                              return
                          else:
                              self.ids.trading_output.text = "Insufficient crypto"
                              return
                  self.ids.trading_output.text = "Crypto not found"
              except requests.exceptions.RequestException as e:
                  self.ids.trading_output.text = f"Error: {e}"

          def update_dashboard(self):
              total_value = self.balance
              try:
                  response = requests.get("http://127.0.0.1:5000/crypto_data")
                  response.raise_for_status()
                  data = response.json()
                  for symbol, amount in self.portfolio.items():
                      for item in data:
                          if item['symbol'].upper() == symbol.upper():
                              total_value += item['current_price'] * amount
                              break
              except requests.exceptions.RequestException as e:
                  self.ids.trading_output.text = f"Error: {e}"
              profit_loss = total_value - 10000
              self.ids.balance_label.text = f"Balance: ${self.balance:.2f}"
              self.ids.portfolio_value_label.text = f"Portfolio Value: ${total_value:.2f}"
              self.ids.profit_loss_label.text = f"Profit/Loss: ${profit_loss:.2f}"
              self.ids.transaction_history.text = "\n".join(self.transactions[-5:])

      class PredictionScreen(Screen):
          prediction_text = StringProperty()
          def __init__(self, **kwargs):
              super().__init__(**kwargs)
              self.prediction_text = "Select a crypto to see prediction"

          def predict(self, symbol):
              threading.Thread(target=self._predict, args=(symbol,)).start()

          def _predict(self, symbol):
              try:
                  response = requests.get(f"http://127.0.0.1:5000/predict/{symbol}")
                  response.raise_for_status()
                  data = response.json()
                  if "prediction" in data:
                      self.prediction_text = f"Predicted price for {symbol}: ${data['prediction']:.2f}"
                  else:
                      self.prediction_text = f"Error: {data.get('error', 'Unknown error')}"
              except requests.exceptions.RequestException as e:
                  self.prediction_text = f"Error: {e}"

      class ContentNavigationDrawer(MDNavigationDrawer):
          pass

      class DrawerList(OneLineListItem):
          def on_release(self):
              self.parent.set_color_item(self)
              app = App.get_running_app()
              app.screen_manager.current = self.text.lower()

      kv = """
      #:import get_color_from_hex kivy.utils.get_color_from_hex

      <ContentNavigationDrawer>:
          ScrollView:
              MDList:
                  DrawerList:
                      text: "Crypto"
                      on_release:
                          app.root.current = "crypto"
                  DrawerList:
                      text: "Trading"
                      on_release:
                          app.root.current = "trading"
                  DrawerList:
                      text: "Prediction"
                      on_release:
                          app.root.current = "prediction"

      <CryptoScreen>:
          MDBoxLayout:
              orientation: 'vertical'
              MDTopAppBar:
                  title: "Crypto Prices"
                  left_action_items: [["menu", lambda x: nav_drawer.set_state("open")]]
              ScrollView:
                  MDList:
                      MDLabel:
                          text: root.crypto_list
                          halign: "center"
                      GridLayout:
                          cols: 2
                          size_hint_y: None
                          height: self.minimum_height
                          padding: 10
                          spacing: 10
                          canvas.before:
                              Color:
                                  rgba: get_color_from_hex("#222222")
                              Rectangle:
                                  pos: self.pos
                                  size: self.size
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          