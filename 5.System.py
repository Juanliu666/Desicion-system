import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="多源混酸回收利用智能决策系统",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
    .recovery-box {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .recovery-box h3 {
        color: white;
        margin: 0;
        font-size: 1.2rem;
    }
    .recovery-box h1 {
        color: white;
        font-size: 4rem;
        margin: 10px 0;
        font-weight: bold;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border: none;
        border-radius: 5px;
        margin: 0 auto;
        display: block;
    }
    .stButton > button:hover {
        background-color: #1565C0;
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .result-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .parameter-label {
        font-weight: bold;
        color: #0D47A1;
    }
    .parameter-value {
        font-size: 1.2rem;
        color: #1E88E5;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #ffc107;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


class PhosphorusRecoverySystem:
    """多源混酸回收利用智能决策系统"""

    def __init__(self, model_path, scaler_path):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.model = None
        self.scaler = None
        self.input_features = ['H3PO4（%）', 'HNO3（%）', 'H2SO4（%）',
                               'Al3+（%）', 'Cu2+（%）', 'Fe3+（%）']
        self.output_features = ['工艺类型', '底液浓度（%）', '添加种类',
                                '添加比例（%）', 'pH值', '反应温度（℃）',
                                '反应时间（h）', '电压（V）', 'P回收率（%）']

        # 定义每个参数的合理范围
        self.param_ranges = {
            '工艺类型': (0, 4),
            '底液浓度（%）': (0, 50),
            '添加种类': (0, 5),
            '添加比例（%）': (0, 100),
            'pH值': (0, 14),
            '反应温度（℃）': (0, 100),
            '反应时间（h）': (0, 48),
            '电压（V）': (0, 12),
            'P回收率（%）': (0, 100)
        }

        # 工艺类型映射
        self.process_types = {
            0: '化学沉淀法',
            1: '电化学法',
            2: '吸附法',
            3: '结晶法',
            4: '膜分离法'
        }

        # 添加种类映射
        self.additives = {
            0: '氯化钙 (CaCl₂)',
            1: '氢氧化钙 (Ca(OH)₂)',
            2: '氧化钙 (CaO)',
            3: '氯化镁 (MgCl₂)',
            4: '氢氧化镁 (Mg(OH)₂)',
            5: '复合添加剂'
        }

    def load_model(self):
        """加载模型和标准化器"""
        try:
            # 检查文件是否存在
            if not self.model_path.exists():
                return False, f"模型文件不存在: {self.model_path}"
            if not self.scaler_path.exists():
                return False, f"标准化器文件不存在: {self.scaler_path}"

            # 加载模型
            try:
                # 尝试用joblib加载模型
                self.model = joblib.load(self.model_path)
                if self.model is None:
                    # 如果joblib失败，尝试pickle
                    with open(self.model_path, 'rb') as f:
                        self.model = pickle.load(f)
            except Exception as e:
                try:
                    with open(self.model_path, 'rb') as f:
                        self.model = pickle.load(f)
                except Exception as e2:
                    return False, f"模型加载失败: {str(e2)}"

            # 加载标准化器
            try:
                # 尝试用joblib加载标准化器
                self.scaler = joblib.load(self.scaler_path)
                if self.scaler is None:
                    # 如果joblib失败，尝试pickle
                    with open(self.scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
            except Exception as e:
                try:
                    with open(self.scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                except Exception as e2:
                    return False, f"标准化器加载失败: {str(e2)}"

            if self.model is not None:
                model_type = type(self.model).__name__
                return True, f"✅ 模型加载成功 - 类型: {model_type}"
            else:
                return False, "❌ 模型加载失败: 模型为空"

        except Exception as e:
            return False, f"❌ 系统初始化失败: {str(e)}"

    def predict(self, input_data):
        """进行预测"""
        try:
            # 转换为numpy数组
            X = np.array(input_data).reshape(1, -1)

            # 标准化（如果有scaler）
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            # 预测
            predictions = self.model.predict(X_scaled)

            # 处理预测结果
            if len(predictions.shape) > 1:
                predictions = predictions[0]

            # 后处理预测结果
            processed_predictions = self.post_process_predictions(predictions)

            # 根据预测值生成详细的推荐参数
            results = self.generate_recommendations(input_data, processed_predictions)

            return results, None, predictions

        except Exception as e:
            return None, str(e), None

    def post_process_predictions(self, predictions):
        """后处理预测结果，确保值在合理范围内"""
        processed = []

        for i, value in enumerate(predictions):
            param_name = self.output_features[i]
            min_val, max_val = self.param_ranges[param_name]

            # 对分类特征进行取整
            if param_name in ['工艺类型', '添加种类']:
                # 取最近的整数，并限制在范围内
                int_value = int(round(value))
                processed.append(max(min_val, min(max_val, int_value)))
            else:
                # 对连续值进行截断
                processed.append(max(min_val, min(max_val, value)))

        return np.array(processed)

    def generate_recommendations(self, input_data, predictions):
        """根据输入和预测生成详细的推荐参数"""

        # 确保predictions是可迭代的
        if np.isscalar(predictions):
            predictions = [predictions] * len(self.output_features)
        else:
            predictions = np.array(predictions).flatten()

        # 创建结果字典
        results = {}
        for i, feature in enumerate(self.output_features):
            if i < len(predictions):
                value = predictions[i]
                # 对分类特征进行处理
                if feature == '工艺类型':
                    value = self.map_process_type(value)
                elif feature == '添加种类':
                    value = self.map_additive_type(value)
                # 数值特征保留两位小数
                elif isinstance(value, (int, float, np.number)):
                    value = round(float(value), 2)
                results[feature] = value

        return results

    def map_process_type(self, value):
        """映射工艺类型"""
        try:
            if isinstance(value, (int, float, np.integer, np.floating)):
                return self.process_types.get(int(value), f'其他工艺({value:.2f})')
            return str(value)
        except:
            return str(value)

    def map_additive_type(self, value):
        """映射添加种类"""
        try:
            if isinstance(value, (int, float, np.integer, np.floating)):
                return self.additives.get(int(value), f'其他添加剂({value:.2f})')
            return str(value)
        except:
            return str(value)

    def validate_input(self, input_data):
        """验证输入数据的合理性"""
        warnings = []

        # 定义各成分的合理范围
        ranges = {
            'H3PO4（%）': (0, 50),
            'HNO3（%）': (0, 30),
            'H2SO4（%）': (0, 40),
            'Al3+（%）': (0, 20),
            'Cu2+（%）': (0, 15),
            'Fe3+（%）': (0, 20)
        }

        for i, feature in enumerate(self.input_features):
            value = input_data[i]
            min_val, max_val = ranges[feature]
            if value < min_val:
                warnings.append(f"⚠️ {feature} 的值 {value}% 低于常见范围 ({min_val}-{max_val}%)")
            elif value > max_val:
                warnings.append(f"⚠️ {feature} 的值 {value}% 高于常见范围 ({min_val}-{max_val}%)")

        # 检查总浓度
        total = sum(input_data)
        if total > 100:
            warnings.append(f"⚠️ 总浓度 {total:.1f}% 超过100%，可能存在输入错误")
        elif total < 1:
            warnings.append(f"⚠️ 总浓度 {total:.1f}% 过低，可能影响回收效果")

        return warnings


def display_results(results):
    """显示结果 - 使用水平布局而不是嵌套列"""

    # 创建两行两列的网格，但不使用st.columns嵌套
    st.markdown('<h2 class="sub-header">💡 智能推荐结果</h2>', unsafe_allow_html=True)

    # 第一行：P回收率
    recovery_rate = results.get('P回收率（%）', 0)

    # 根据回收率高低显示不同颜色
    if recovery_rate >= 80:
        recovery_color = "linear-gradient(135deg, #28a745 0%, #20c997 100%)"
    elif recovery_rate >= 50:
        recovery_color = "linear-gradient(135deg, #ffc107 0%, #fd7e14 100%)"
    else:
        recovery_color = "linear-gradient(135deg, #dc3545 0%, #c82333 100%)"

    st.markdown(
        f"""
        <div style="text-align: center; padding: 20px; background: {recovery_color}; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: white; margin: 0;">预计磷回收率</h3>
            <h1 style="color: white; font-size: 4rem; margin: 10px 0; font-weight: bold;">{recovery_rate}%</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 第二行：工艺参数 - 使用自定义HTML/CSS布局，避免st.columns
    st.markdown("##### 🏭 工艺参数")

    # 使用HTML创建两列布局，避免Streamlit的列嵌套
    col1_html = f"""
    <div style="display: flex; gap: 20px; margin-bottom: 20px;">
        <div style="flex: 1; background-color: #f0f2f6; padding: 15px; border-radius: 8px;">
            <div style="font-weight: bold; color: #0D47A1;">工艺类型</div>
            <div style="font-size: 1.2rem; color: #1E88E5;">{results.get('工艺类型', 'N/A')}</div>
        </div>
        <div style="flex: 1; background-color: #f0f2f6; padding: 15px; border-radius: 8px;">
            <div style="font-weight: bold; color: #0D47A1;">底液浓度</div>
            <div style="font-size: 1.2rem; color: #1E88E5;">{results.get('底液浓度（%）', 0)}%</div>
        </div>
    </div>
    <div style="display: flex; gap: 20px; margin-bottom: 20px;">
        <div style="flex: 1; background-color: #f0f2f6; padding: 15px; border-radius: 8px;">
            <div style="font-weight: bold; color: #0D47A1;">添加种类</div>
            <div style="font-size: 1.2rem; color: #1E88E5;">{results.get('添加种类', 'N/A')}</div>
        </div>
        <div style="flex: 1; background-color: #f0f2f6; padding: 15px; border-radius: 8px;">
            <div style="font-weight: bold; color: #0D47A1;">添加比例</div>
            <div style="font-size: 1.2rem; color: #1E88E5;">{results.get('添加比例（%）', 0)}%</div>
        </div>
    </div>
    """
    st.markdown(col1_html, unsafe_allow_html=True)

    # 第三行：操作条件
    st.markdown("##### ⚙️ 操作条件")

    col2_html = f"""
    <div style="display: flex; gap: 20px; margin-bottom: 20px;">
        <div style="flex: 1; background-color: #f0f2f6; padding: 15px; border-radius: 8px;">
            <div style="font-weight: bold; color: #0D47A1;">pH值</div>
            <div style="font-size: 1.2rem; color: #1E88E5;">{results.get('pH值', 0)}</div>
        </div>
        <div style="flex: 1; background-color: #f0f2f6; padding: 15px; border-radius: 8px;">
            <div style="font-weight: bold; color: #0D47A1;">反应温度</div>
            <div style="font-size: 1.2rem; color: #1E88E5;">{results.get('反应温度（℃）', 0)}°C</div>
        </div>
    </div>
    <div style="display: flex; gap: 20px; margin-bottom: 20px;">
        <div style="flex: 1; background-color: #f0f2f6; padding: 15px; border-radius: 8px;">
            <div style="font-weight: bold; color: #0D47A1;">反应时间</div>
            <div style="font-size: 1.2rem; color: #1E88E5;">{results.get('反应时间（h）', 0)} h</div>
        </div>
        <div style="flex: 1; background-color: #f0f2f6; padding: 15px; border-radius: 8px;">
            <div style="font-weight: bold; color: #0D47A1;">电压</div>
            <div style="font-size: 1.2rem; color: #1E88E5;">{results.get('电压（V）', 0)} V</div>
        </div>
    </div>
    """
    st.markdown(col2_html, unsafe_allow_html=True)


def main():
    # 标题
    st.markdown('<h1 class="main-header">⚗️ 多源混酸回收利用智能决策系统</h1>', unsafe_allow_html=True)

    # 侧边栏 - 只显示系统信息
    with st.sidebar:
        st.markdown("## 系统控制面板")
        st.markdown("---")
        st.markdown("### 关于系统")
        st.info(
            "本系统基于SVR机器学习模型，根据废液特性自动推荐"
            "最佳工艺类型和操作参数，并预测磷回收率。"
        )

        # 调试模式开关
        st.session_state.debug_mode = st.checkbox("🔧 调试模式", value=False)

        # 显示系统状态
        st.markdown("---")
        st.markdown("### 系统状态")
        if st.session_state.get('model_loaded', False):
            st.success("✅ 模型已加载")

    # 初始化系统并加载模型
    if 'system' not in st.session_state:
        # 获取当前脚本所在目录
        current_dir = Path(__file__).parent
        model_path = current_dir / "SVR模型.pkl"
        scaler_path = current_dir / "标准化器.pkl"

        st.session_state.system = PhosphorusRecoverySystem(model_path, scaler_path)

        # 自动加载模型
        with st.spinner("正在加载模型和标准化器..."):
            success, message = st.session_state.system.load_model()
            if success:
                st.session_state.model_loaded = True
                # 在侧边栏显示成功信息，不在主界面显示
            else:
                st.session_state.model_loaded = False
                st.error(f"系统初始化失败: {message}")
                st.stop()

    # 主界面
    if not st.session_state.get('model_loaded', False):
        st.error("系统模型加载失败，请检查模型文件是否存在")
        with st.expander("📖 故障排除"):
            st.markdown("""
            ### 请检查以下文件是否在程序同目录下：
            - `SVR模型.pkl` - SVR模型文件
            - `标准化器.pkl` - 标准化器文件

            如果文件缺失，请确保这两个文件与程序在同一文件夹中。
            """)
    else:
        # 输入部分
        st.markdown('<h2 class="sub-header">📊 输入废液特性</h2>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="input-section">', unsafe_allow_html=True)

            # 酸浓度输入
            st.markdown("##### 🧪 酸浓度 (%)")

            # 创建三列但不嵌套
            acid_c1, acid_c2, acid_c3 = st.columns(3)
            with acid_c1:
                h3po4 = st.number_input(
                    "H₃PO₄",
                    min_value=0.0, max_value=100.0, value=15.0, step=0.1,
                    format="%.1f",
                    key="h3po4_input"
                )
            with acid_c2:
                hno3 = st.number_input(
                    "HNO₃",
                    min_value=0.0, max_value=100.0, value=8.0, step=0.1,
                    format="%.1f",
                    key="hno3_input"
                )
            with acid_c3:
                h2so4 = st.number_input(
                    "H₂SO₄",
                    min_value=0.0, max_value=100.0, value=12.0, step=0.1,
                    format="%.1f",
                    key="h2so4_input"
                )

            # 金属离子输入
            st.markdown("##### ⚛️ 金属离子浓度 (%)")

            # 创建三列但不嵌套
            metal_c1, metal_c2, metal_c3 = st.columns(3)
            with metal_c1:
                al = st.number_input(
                    "Al³⁺",
                    min_value=0.0, max_value=50.0, value=3.5, step=0.1,
                    format="%.1f",
                    key="al_input"
                )
            with metal_c2:
                cu = st.number_input(
                    "Cu²⁺",
                    min_value=0.0, max_value=50.0, value=2.0, step=0.1,
                    format="%.1f",
                    key="cu_input"
                )
            with metal_c3:
                fe = st.number_input(
                    "Fe³⁺",
                    min_value=0.0, max_value=50.0, value=4.5, step=0.1,
                    format="%.1f",
                    key="fe_input"
                )

            # 收集输入数据
            input_data = [h3po4, hno3, h2so4, al, cu, fe]

            # 显示总浓度
            total_concentration = sum(input_data)
            st.markdown(f"**总浓度:** {total_concentration:.1f}%")

            st.markdown('</div>', unsafe_allow_html=True)

        # 预测按钮 - 居中显示
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_clicked = st.button("🔮 开始智能推荐")

        # 处理预测
        if predict_clicked:
            # 验证输入
            warnings = st.session_state.system.validate_input(input_data)

            if warnings:
                for warning in warnings:
                    st.warning(warning)

            # 进行预测
            with st.spinner("正在分析废液特性，生成推荐方案..."):
                results, error, raw_predictions = st.session_state.system.predict(input_data)

            if error:
                st.error(f"预测失败: {error}")
            elif results:
                st.session_state.last_results = results
                st.session_state.input_data = input_data
                st.session_state.raw_predictions = raw_predictions
                st.success("✓ 推荐生成成功！")

        st.markdown("---")

        # 显示调试信息（如果开启调试模式）
        if st.session_state.get('debug_mode', False) and hasattr(st.session_state, 'raw_predictions'):
            with st.expander("🔧 调试信息 - 原始预测值"):
                raw_data = st.session_state.raw_predictions

                # 创建DataFrame显示原始预测值
                debug_df = pd.DataFrame(
                    [raw_data],
                    columns=st.session_state.system.output_features
                )
                st.dataframe(debug_df)

                # 显示处理后的值对比
                st.markdown("##### 处理后结果")
                processed_df = pd.DataFrame(
                    [list(st.session_state.last_results.values())],
                    columns=st.session_state.system.output_features
                )
                st.dataframe(processed_df)

        # 结果显示部分
        if hasattr(st.session_state, 'last_results'):
            # 直接显示结果
            display_results(st.session_state.last_results)


if __name__ == "__main__":
    main()