import subprocess
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
import sys
from datetime import datetime
warnings.filterwarnings('ignore')


class DriverPriceOptimizer:
    def __init__(self, log_file=None):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.log_file = log_file

    def log(self, message):
        """Запись сообщения в лог-файл и вывод в консоль"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')

    def load_data(self, file_path):
        """Загрузка данных из CSV файла"""
        self.log("📊 Загрузка данных...")
        self.df = pd.read_csv(file_path)

        # Предобработка данных
        self.df['is_done'] = self.df['is_done'].map({'done': 1, 'cancel': 0})
        self.df['order_timestamp'] = pd.to_datetime(self.df['order_timestamp'])
        self.df['tender_timestamp'] = pd.to_datetime(self.df['tender_timestamp'])

        self.log(f"✅ Загружено {len(self.df)} записей")
        self.log(f"📈 Процент принятых заказов: {self.df['is_done'].mean():.1%}")

        return self.df

    def create_features(self):
        """Создание признаков для ML модели"""
        self.log("🔧 Создание признаков...")

        # Временные признаки
        self.df['order_hour'] = self.df['order_timestamp'].dt.hour
        self.df['order_dayofweek'] = self.df['order_timestamp'].dt.dayofweek
        self.df['order_month'] = self.df['order_timestamp'].dt.month
        self.df['is_weekend'] = (self.df['order_dayofweek'] >= 5).astype(int)
        self.df['is_night'] = ((self.df['order_hour'] >= 23) | (self.df['order_hour'] <= 6)).astype(int)
        self.df['is_rush_hour'] = ((self.df['order_hour'] >= 7) & (self.df['order_hour'] <= 10) |
                                   (self.df['order_hour'] >= 17) & (self.df['order_hour'] <= 20)).astype(int)

        # Признаки цены
        self.df['price_ratio'] = self.df['price_bid_local'] / self.df['price_start_local']
        self.df['price_diff'] = self.df['price_bid_local'] - self.df['price_start_local']
        self.df['price_diff_percent'] = (self.df['price_diff'] / self.df['price_start_local']) * 100

        # Время ответа водителя
        self.df['response_time_sec'] = (self.df['tender_timestamp'] - self.df['order_timestamp']).dt.total_seconds()

        # Кодирование категориальных переменных
        categorical_cols = ['carmodel', 'carname', 'platform']
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].fillna('unknown'))
                self.label_encoders[col] = le

        self.log("✅ Признаки созданы")
        return self.df

    def prepare_model_data(self):
        """Подготовка данных для обучения модели"""
        feature_cols = [
            # Ценовые признаки
            'price_start_local', 'price_ratio', 'price_diff', 'price_diff_percent',
            # Временные признаки
            'order_hour', 'order_dayofweek', 'order_month', 'is_weekend', 'is_night', 'is_rush_hour',
            'response_time_sec',
            # Рейтинги
            'driver_rating',
            # Закодированные категориальные
            'carmodel_encoded', 'carname_encoded', 'platform_encoded'
        ]

        # Фильтруем только существующие колонки
        self.feature_cols = [col for col in feature_cols if col in self.df.columns]
        self.log(f"✅ Используется {len(self.feature_cols)} признаков")

        return self.feature_cols

    def train_model(self, test_size=0.2):
        """Обучение ML модели"""
        self.log("🤖 Обучение модели...")

        X = self.df[self.feature_cols]
        y = self.df['is_done']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])

        self.model = GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42
        )

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = np.mean(y_pred == y_test)

        self.log("📊 Модель обучена успешно!")
        self.log(f"🎯 Точность предсказаний: {accuracy:.1%}")
        self.log(f"📈 ROC-AUC: {roc_auc:.3f}")

        self.feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Логируем важность признаков
        self.log("\n🔍 ВАЖНОСТЬ ПРИЗНАКОВ:")
        self.log("-" * 40)
        for _, row in self.feature_importance.head(10).iterrows():
            self.log(f"{row['feature']}: {row['importance']:.4f}")

        return self.model

    def predict_acceptance_probability(self, order_features, bid_price):
        """Предсказание вероятности принятия цены"""
        features_df = order_features.copy()
        price_start = features_df['price_start_local'].iloc[0]

        # Обновляем ценовые признаки
        features_df['price_bid_local'] = bid_price
        features_df['price_ratio'] = bid_price / price_start
        features_df['price_diff'] = bid_price - price_start
        features_df['price_diff_percent'] = ((bid_price - price_start) / price_start) * 100

        # Подготовка признаков
        X = features_df[self.feature_cols]
        X_scaled = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X_scaled[numerical_cols] = self.scaler.transform(X[numerical_cols])

        # Предсказание вероятности
        probability = self.model.predict_proba(X_scaled)[0][1]

        return probability

    def get_price_recommendations(self, order_features, price_start):
        """Получение готовых рекомендаций по ценам"""

        # Рассчитываем оптимальные цены для каждой стратегии
        prices_to_test = []

        # Консервативная стратегия: цены близкие к исходной
        conservative_prices = [
            int(price_start * 1.0),  # +0%
            int(price_start * 1.1),  # +10%
            int(price_start * 1.2),  # +20%
            int(price_start * 1.3)  # +30%
        ]

        # Сбалансированная стратегия: средний диапазон
        balanced_prices = [
            int(price_start * 1.2),  # +20%
            int(price_start * 1.4),  # +40%
            int(price_start * 1.6),  # +60%
            int(price_start * 1.8)  # +80%
        ]

        # Агрессивная стратегия: высокие цены
        aggressive_prices = [
            int(price_start * 1.5),  # +50%
            int(price_start * 1.8),  # +80%
            int(price_start * 2.0),  # +100%
            int(price_start * 2.2)  # +120%
        ]

        # Находим лучшую цену для каждой стратегии
        conservative_results = []
        for price in conservative_prices:
            prob = self.predict_acceptance_probability(order_features, price)
            income = price * prob
            conservative_results.append({'price': price, 'prob': prob, 'income': income})

        balanced_results = []
        for price in balanced_prices:
            prob = self.predict_acceptance_probability(order_features, price)
            income = price * prob
            balanced_results.append({'price': price, 'prob': prob, 'income': income})

        aggressive_results = []
        for price in aggressive_prices:
            prob = self.predict_acceptance_probability(order_features, price)
            income = price * prob
            aggressive_results.append({'price': price, 'prob': prob, 'income': income})

        # Выбираем лучшие варианты
        conservative_best = max(conservative_results, key=lambda x: x['income'])
        balanced_best = max(balanced_results, key=lambda x: x['income'])
        aggressive_best = max(aggressive_results, key=lambda x: x['income'])

        recommendations = {
            'conservative': {
                'price': conservative_best['price'],
                'probability': conservative_best['prob'],
                'expected_income': conservative_best['income'],
                'markup': conservative_best['price'] - price_start,
                'markup_percent': ((conservative_best['price'] - price_start) / price_start) * 100
            },
            'balanced': {
                'price': balanced_best['price'],
                'probability': balanced_best['prob'],
                'expected_income': balanced_best['income'],
                'markup': balanced_best['price'] - price_start,
                'markup_percent': ((balanced_best['price'] - price_start) / price_start) * 100
            },
            'aggressive': {
                'price': aggressive_best['price'],
                'probability': aggressive_best['prob'],
                'expected_income': aggressive_best['income'],
                'markup': aggressive_best['price'] - price_start,
                'markup_percent': ((aggressive_best['price'] - price_start) / price_start) * 100
            }
        }

        return recommendations

    def save_model(self, filepath='driver_price_predictor.pkl'):
        """Сохранение модели"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_cols': self.feature_cols,
            'feature_importance': self.feature_importance
        }

        joblib.dump(model_data, filepath)
        self.log(f"💾 Модель сохранена: {filepath}")

    def load_model(self, filepath):
        """Загрузка модели"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_cols = model_data['feature_cols']
        self.feature_importance = model_data['feature_importance']
        self.log(f"📂 Модель загружена: {filepath}")


def run_complete_pipeline(csv_file_path, log_file_path):
    """Запуск полного пайплайна"""
    
    # Создаем лог-файл и записываем заголовок
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("🚀 СИСТЕМА ОПТИМИЗАЦИИ ЦЕН ДЛЯ ВОДИТЕЛЕЙ\n")
        f.write("=" * 50 + "\n")
        f.write(f"Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Файл с данными: {csv_file_path}\n")
        f.write("=" * 50 + "\n\n")

    optimizer = DriverPriceOptimizer(log_file=log_file_path)

    try:
        # Обучение модели
        optimizer.log("🚀 ЗАПУСК СИСТЕМЫ ОПТИМИЗАЦИИ ЦЕН")
        optimizer.log("=" * 45)

        optimizer.load_data(csv_file_path)
        optimizer.create_features()
        optimizer.prepare_model_data()
        optimizer.train_model()

        optimizer.log("\n" + "🎯 РЕКОМЕНДАЦИИ ДЛЯ ВОДИТЕЛЕЙ")
        optimizer.log("=" * 45)

        # Тестирование на нескольких заказах
        sample_orders = optimizer.df.sample(min(3, len(optimizer.df)))

        for i, (idx, order) in enumerate(sample_orders.iterrows(), 1):
            optimizer.log(f"\n📦 ЗАКАЗ #{i}")
            optimizer.log("─" * 30)
            optimizer.log(f"💰 Цена пассажира: {order['price_start_local']} ₽")
            optimizer.log(f"⭐ Рейтинг водителя: {order['driver_rating']}")
            optimizer.log(f"🕐 Время: {order['order_hour']}:00")

            # Получаем рекомендации
            order_data = optimizer.df.loc[idx:idx][optimizer.feature_cols].copy()
            recommendations = optimizer.get_price_recommendations(order_data, order['price_start_local'])

            optimizer.log("\n🎪 СТРАТЕГИИ ЦЕНООБРАЗОВАНИЯ:")
            optimizer.log("─" * 45)

            # Консервативная стратегия
            cons = recommendations['conservative']
            optimizer.log(f"\n🟢 КОНСЕРВАТИВНАЯ")
            optimizer.log(f"   💰 Рекомендуемая цена: {cons['price']} ₽")
            optimizer.log(f"   📈 Вероятность принятия: {cons['probability']:.1%}")
            optimizer.log(f"   🎯 Ожидаемый доход: {cons['expected_income']:.0f} ₽")
            optimizer.log(f"   📊 Надбавка: +{cons['markup']} ₽ (+{cons['markup_percent']:.0f}%)")

            # Сбалансированная стратегия
            bal = recommendations['balanced']
            optimizer.log(f"\n🟡 СБАЛАНСИРОВАННАЯ")
            optimizer.log(f"   💰 Рекомендуемая цена: {bal['price']} ₽")
            optimizer.log(f"   📈 Вероятность принятия: {bal['probability']:.1%}")
            optimizer.log(f"   🎯 Ожидаемый доход: {bal['expected_income']:.0f} ₽")
            optimizer.log(f"   📊 Надбавка: +{bal['markup']} ₽ (+{bal['markup_percent']:.0f}%)")

            # Агрессивная стратегия
            agg = recommendations['aggressive']
            optimizer.log(f"\n🔴 АГРЕССИВНАЯ")
            optimizer.log(f"   💰 Рекомендуемая цена: {agg['price']} ₽")
            optimizer.log(f"   📈 Вероятность принятия: {agg['probability']:.1%}")
            optimizer.log(f"   🎯 Ожидаемый доход: {agg['expected_income']:.0f} ₽")
            optimizer.log(f"   📊 Надбавка: +{agg['markup']} ₽ (+{agg['markup_percent']:.0f}%)")

            optimizer.log("\n" + "💡 РЕКОМЕНДАЦИЯ:")
            optimizer.log("─" * 20)
            best_strategy = max(recommendations.items(), key=lambda x: x[1]['expected_income'])
            strategy_name = {
                'conservative': 'консервативную',
                'balanced': 'сбалансированную',
                'aggressive': 'агрессивную'
            }[best_strategy[0]]

            optimizer.log(f"Выберите {strategy_name} стратегию:")
            optimizer.log(f"💰 Установите цену: {best_strategy[1]['price']} ₽")
            optimizer.log(f"🎯 Ожидаемый доход: {best_strategy[1]['expected_income']:.0f} ₽")

        # Сохранение модели
        optimizer.save_model()

        optimizer.log("\n" + "=" * 45)
        optimizer.log("✅ СИСТЕМА ГОТОВА К РАБОТЕ!")
        optimizer.log("=" * 45)
        
        optimizer.log(f"\n📁 Все результаты сохранены в файл: {log_file_path}")

        return optimizer

    except Exception as e:
        optimizer.log(f"❌ Ошибка: {e}")
        return None

# ЗАПУСК ПРОГРАММЫ
if __name__ == "__main__":
    CSV_FILE_PATH = "train.csv"
    LOG_FILE_PATH = f"price_optimization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"    
    print(f"📝 Логирование будет сохранено в файл: {LOG_FILE_PATH}")    
    optimizer = run_complete_pipeline(CSV_FILE_PATH, LOG_FILE_PATH)
