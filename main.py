import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Заголовок приложения
st.set_page_config(page_title="Монте-Карло симулятор торговой стратегии", layout="wide")
st.title("📈 Монте-Карло симулятор торговой стратегии")
st.markdown("""
Анализ потенциальной прибыльности торговой стратегии с использованием метода Монте-Карло.
Настройте параметры ниже и запустите симуляцию.
""")

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Параметры симуляции")
    
    # Основные параметры
    weeks = st.slider("Количество недель", min_value=1, max_value=260, value=52, step=1,
                     help="Общий период симуляции в неделях")
    
    # Поле для ввода количества сделок
    deals_input = st.text_input("Количество сделок в неделю (через запятую)", "2, 5, 10",
                               help="Введите несколько значений через запятую для сравнения")
    
    # Парсинг ввода
    try:
        deals_per_week = [int(x.strip()) for x in deals_input.split(',') if x.strip().isdigit()]
        if not deals_per_week:
            st.error("Пожалуйста, введите хотя бы одно число")
            deals_per_week = [2, 5, 10]
    except:
        st.error("Некорректный ввод. Используются значения по умолчанию: 2, 5, 10")
        deals_per_week = [2, 5, 10]
    
    # Отображение текущих значений
    st.caption(f"Будут проанализированы: {deals_per_week} сделок/неделю")
    
    win_rate = st.slider("Винрейт (вероятность успешной сделки)", min_value=0.0, max_value=1.0, 
                         value=0.8, step=0.01, format="%.2f",
                         help="Процент успешных сделок")
    
    avg_profit = st.slider("Средний профит с успешной сделки (коэффициент)", min_value=1.01, 
                          max_value=3.0, value=1.35, step=0.01, format="%.2f",
                          help="Например: 1.35 означает 35% прибыли от успешной сделки")
    
    n_simulations = st.number_input("Количество симуляций", min_value=100, max_value=100000, 
                                   value=10_000, step=1_000,
                                   help="Чем больше симуляций, тем точнее результаты")
    
    # Кнопка запуска
    run_simulation = st.button("🚀 Запустить симуляцию", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📊 Пояснения")
    st.markdown("""
    - **Винрейт**: процент выигрышных сделок
    - **Средний профит**: прибыль от успешной сделки (1.35 = +35%)
    - **Количество симуляций**: больше = точнее, но медленнее
    """)

# Функция симуляции (оригинальная)
def monte_carlo_simulation_with_history(
    weeks: int = 52,
    deals_per_week: List[int] = [2, 5, 10],
    win_rate: float = 0.8,
    avg_profit: float = 1.35,
    n_simulations: int = 1000,
    save_history: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Монте-Карло симуляция торговой стратегии с сохранением истории для Plotly
    """
    profit_per_win = avg_profit - 1
    
    all_simulations_results = []
    history_dict = {} if save_history else None
    
    for cnt_deal in deals_per_week:
        # Векторизованная генерация всех сделок
        deals_all = np.random.choice(
            [0, 1], 
            size=(n_simulations, weeks, cnt_deal), 
            p=[1 - win_rate, win_rate]
        )
        
        # Расчет прибыли
        losses_all = (deals_all == 0).sum(axis=2) * -1
        wins_all = (deals_all == 1).sum(axis=2) * profit_per_win
        weekly_profits_all = losses_all + wins_all
        cumulative_profits_all = np.cumsum(weekly_profits_all, axis=1)
        final_profits = cumulative_profits_all[:, -1]
        
        # Сохраняем историю для графиков
        if save_history:
            median_cumulative = np.percentile(cumulative_profits_all, 50, axis=0)
            p5_cumulative = np.percentile(cumulative_profits_all, 5, axis=0)
            p25_cumulative = np.percentile(cumulative_profits_all, 25, axis=0)
            p75_cumulative = np.percentile(cumulative_profits_all, 75, axis=0)
            p95_cumulative = np.percentile(cumulative_profits_all, 95, axis=0)
            
            history_dict[cnt_deal] = {
                'median': np.round(median_cumulative, 2),
                'p5': np.round(p5_cumulative, 2),
                'p25': np.round(p25_cumulative, 2),
                'p75': np.round(p75_cumulative, 2),
                'p95': np.round(p95_cumulative, 2),
                'weeks': list(range(1, weeks + 1)),
                'all_cumulative': np.round(cumulative_profits_all, 2),
                'final_profits': np.round(final_profits, 2)
            }
        
        # Статистика
        mean_profit = np.mean(final_profits)
        median_profit = np.median(final_profits)
        std_profit = np.std(final_profits)
        percentile_5 = np.percentile(final_profits, 5)
        percentile_95 = np.percentile(final_profits, 95)
        probability_loss = np.mean(final_profits < 0) * 100
        sharpe_ratio = mean_profit / std_profit if std_profit > 0 else 0
        
        all_simulations_results.append({
            'deals_per_week': cnt_deal,
            'win_rate': round(win_rate, 2),
            'avg_profit': round(avg_profit, 2),
            'mean_profit': round(mean_profit, 2),
            'median_profit': round(median_profit, 2),
            'std_profit': round(std_profit, 2),
            'percentile_5': round(percentile_5, 2),
            'percentile_95': round(percentile_95, 2),
            'probability_loss_%': round(probability_loss, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'expected_profit': round(cnt_deal * weeks * ((win_rate * profit_per_win) - ((1 - win_rate) * 1)), 2)
        })
    
    results_df = pd.DataFrame(all_simulations_results)
    results_df['profit_per_deal'] = np.round(results_df['mean_profit'] / (results_df['deals_per_week'] * weeks), 2)
    results_df['mean_weekly_profit'] = np.round(results_df['mean_profit'] / weeks, 2)
    
    # Округляем все числовые колонки в DataFrame
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    results_df[numeric_cols] = results_df[numeric_cols].round(2)
    
    return results_df, history_dict


def create_plots(results_df: pd.DataFrame, history_dict: Dict):
    """Создание графиков на основе результатов симуляции"""
    
    # 1. График кумулятивной прибыли
    fig1 = go.Figure()
    
    # Используем стандартные цвета Plotly
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (cnt_deal, history) in enumerate(history_dict.items()):
        color = colors[i % len(colors)]
        
        # Преобразуем HEX в RGB
        if color.startswith('#'):
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            rgba_light = f'rgba({r}, {g}, {b}, 0.2)'
            rgba_very_light = f'rgba({r}, {g}, {b}, 0.1)'
        else:
            # Если цвет не в HEX формате, используем fallback
            rgba_light = 'rgba(31, 119, 180, 0.2)'
            rgba_very_light = 'rgba(31, 119, 180, 0.1)'
        
        # Основная область (между p25 и p75)
        fig1.add_trace(go.Scatter(
            x=history['weeks'] + history['weeks'][::-1],
            y=list(history['p75']) + list(history['p25'])[::-1],
            fill='toself',
            fillcolor=rgba_light,
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name=f'{cnt_deal} сделок/неделю (25-75%)'
        ))
        
        # Область между p5 и p95
        fig1.add_trace(go.Scatter(
            x=history['weeks'] + history['weeks'][::-1],
            y=list(history['p95']) + list(history['p5'])[::-1],
            fill='toself',
            fillcolor=rgba_very_light,
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name=f'{cnt_deal} сделок/неделю (5-95%)'
        ))
        
        # Медиана
        fig1.add_trace(go.Scatter(
            x=history['weeks'],
            y=history['median'],
            mode='lines',
            line=dict(color=color, width=2),
            name=f'{cnt_deal} сделок/неделю (медиана)'
        ))
    
    fig1.update_layout(
        title='Кумулятивная прибыль по неделям (с доверительными интервалами)',
        xaxis_title='Недели',
        yaxis_title='Прибыль',
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # 2. График распределения конечной прибыли
    fig2 = go.Figure()
    
    for i, (cnt_deal, history) in enumerate(history_dict.items()):
        color = colors[i % len(colors)]
        fig2.add_trace(go.Violin(
            y=history['final_profits'],
            name=f'{cnt_deal} сделок/неделю',
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            line_color=color,
            opacity=0.6
        ))
    
    fig2.update_layout(
        title='Распределение конечной прибыли по всем симуляциям',
        xaxis_title='Количество сделок в неделю',
        yaxis_title='Конечная прибыль',
        height=500
    )
    
    # 3. Heatmap корреляции между параметрами
    # Выбираем только числовые столбцы для корреляции
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    # Исключаем колонки, которые не хотим анализировать
    exclude_cols = ['deals_per_week', 'win_rate', 'avg_profit']
    heatmap_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(heatmap_cols) > 1:
        corr_matrix = results_df[heatmap_cols].corr()
        # Округляем корреляционную матрицу
        corr_matrix = corr_matrix.round(2)
        fig3 = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu',
            title='Корреляция между метриками',
            labels=dict(color="Корреляция")
        )
        fig3.update_layout(height=400)
    else:
        fig3 = go.Figure()
        fig3.add_annotation(
            text="Недостаточно данных для корреляции",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig3.update_layout(height=400, title="Корреляция между метриками")
    
    # 4. Bar chart с основными метриками
    fig4 = go.Figure()
    
    metrics = ['mean_profit', 'median_profit', 'sharpe_ratio']
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    if available_metrics:
        for i, metric in enumerate(available_metrics):
            color = colors[i % len(colors)]
            # Используем уже округленные значения из results_df
            y_values = results_df[metric]
            fig4.add_trace(go.Bar(
                x=results_df['deals_per_week'],
                y=y_values,
                name=metric.replace('_', ' ').title(),
                text=y_values.round(2),
                textposition='outside',
                marker_color=color
            ))
        
        fig4.update_layout(
            title='Основные метрики по количеству сделок',
            xaxis_title='Сделок в неделю',
            yaxis_title='Значение',
            barmode='group',
            height=500
        )
    else:
        fig4 = go.Figure()
        fig4.add_annotation(
            text="Нет данных для построения графика",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig4.update_layout(
            title='Основные метрики по количеству сделок',
            xaxis_title='Сделок в неделю',
            yaxis_title='Значение',
            height=500
        ) 
     
    return fig1, fig2, fig3, fig4

# Основная часть приложения
if run_simulation:
    with st.spinner('Выполняю симуляцию... Это может занять несколько секунд'):
        # Запуск симуляции
        results_df, history_dict = monte_carlo_simulation_with_history(
            weeks=weeks,
            deals_per_week=deals_per_week,
            win_rate=win_rate,
            avg_profit=avg_profit,
            n_simulations=n_simulations,
            save_history=True
        )
    
    # Отображение результатов
    st.success(f'✅ Симуляция завершена! Проанализировано {n_simulations:,} сценариев.')
    
    # Основные метрики
    st.subheader("📊 Ключевые показатели")
    
    # Создаем колонки для метрик
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_deal = results_df.loc[results_df['mean_profit'].idxmax()]
        st.metric(
            label="Лучшее кол-во сделок",
            value=f"{int(best_deal['deals_per_week'])}/неделю",
            delta=f"{best_deal['mean_profit']:.2f} прибыли"
        )
    
    with col2:
        max_profit = results_df['mean_profit'].max()
        st.metric(
            label="Максимальная средняя прибыль",
            value=f"{max_profit:.2f}",
            delta="за период"
        )
    
    with col3:
        min_loss_prob = results_df['probability_loss_%'].min()
        st.metric(
            label="Минимальная вероятность убытка",
            value=f"{min_loss_prob:.2f}%"
        )
    
    with col4:
        max_sharpe = results_df['sharpe_ratio'].max()
        st.metric(
            label="Лучший коэффициент Шарпа",
            value=f"{max_sharpe:.2f}"
        )
    
    # Детальная таблица результатов
    st.subheader("📋 Детальные результаты")
    
    # Форматирование таблицы
    display_df = results_df.copy().round(2)
    
    # Создаем словарь для форматирования колонок
    format_dict = {}
    for col in display_df.columns:
        if display_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            format_dict[col] = "{:.2f}".format
    
    display_df = display_df.rename(columns=lambda x: x.replace('_', ' ').title())
    
    # Выделяем лучшие значения цветом
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: lightgreen' if v else '' for v in is_max]
    
    def highlight_min(s):
        is_min = s == s.min()
        return ['background-color: lightcoral' if v else '' for v in is_min]
    
    # Применяем стили и форматирование
    styled_df = display_df.style.format(format_dict)\
                                .apply(highlight_max, subset=['Mean Profit', 'Sharpe Ratio'])\
                                .apply(highlight_min, subset=['Probability Loss %', 'Std Profit'])
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Визуализации
    st.subheader("📈 Визуализация результатов")
    
    # Создаем графики
    fig1, fig2, fig3, fig4 = create_plots(results_df, history_dict)
    
    # Отображаем графики
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Кумулятивная прибыль", 
        "📉 Распределение", 
        "🔗 Корреляции",
        "📊 Сравнение метрик"
    ])
    
    with tab1:
        st.plotly_chart(fig1, use_container_width=True)
        
    with tab2:
        st.plotly_chart(fig2, use_container_width=True)
        
    with tab3:
        st.plotly_chart(fig3, use_container_width=True)
        
    with tab4:
        st.plotly_chart(fig4, use_container_width=True)
    
    # Дополнительная информация
    with st.expander("📝 Интерпретация результатов"):
        st.markdown("""
        ### Как читать результаты:
        
        **Кумулятивная прибыль:**
        - **Темные линии** - медианные значения (50-й процентиль)
        - **Светлые области** - диапазон 25-75% процентилей
        - **Очень светлые области** - диапазон 5-95% процентилей
        
        **Коэффициент Шарпа:**
        - > 1.0 - хороший результат
        - > 2.0 - отличный результат
        - < 1.0 - высокий риск относительно доходности
        
        **Вероятность убытка:**
        - < 5% - очень низкий риск
        - 5-15% - умеренный риск
        - > 15% - высокий риск
        """)
    
    # Кнопка для скачивания результатов
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Скачать результаты в CSV",
        data=csv,
        file_name=f"monte_carlo_results_{weeks}weeks_{win_rate}wr_{avg_profit}profit.csv",
        mime="text/csv"
    )

else:
    # Инструкция при первом запуске
    st.info("👈 Настройте параметры в боковой панели и нажмите 'Запустить симуляцию'")
    
    # Пример результатов
    with st.expander("📋 Пример формата результатов"):
        example_df = pd.DataFrame({
            'Сделок в неделю': [2, 5, 10],
            'Средняя прибыль': [36.20, 90.50, 181.00],
            'Медианная прибыль': [35.80, 89.50, 179.00],
            'Вероятность убытка %': [5.20, 1.80, 0.30],
            'Коэффициент Шарпа': [1.25, 1.78, 2.12]
        })
        st.dataframe(example_df, use_container_width=True)
    
    # Быстрые примеры настроек
    st.subheader("⚡ Быстрые пресеты")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Консервативная стратегия", use_container_width=True):
            st.session_state.update({
                "weeks": 26,
                "deals_input": "1, 2, 3",
                "win_rate": 0.85,
                "avg_profit": 1.20,
                "n_simulations": 2000
            })
            st.rerun()
    
    with col2:
        if st.button("Агрессивная стратегия", use_container_width=True):
            st.session_state.update({
                "weeks": 104,
                "deals_input": "10, 20, 30",
                "win_rate": 0.60,
                "avg_profit": 1.80,
                "n_simulations": 5000
            })
            st.rerun()
    
    with col3:
        if st.button("Сбалансированная стратегия", use_container_width=True):
            st.session_state.update({
                "weeks": 52,
                "deals_input": "5, 10, 15",
                "win_rate": 0.75,
                "avg_profit": 1.50,
                "n_simulations": 3000
            })
            st.rerun()

# Футер
st.markdown("---")
st.caption("""
**Примечание:** Результаты симуляции являются теоретическими и не гарантируют реальную доходность. 
Инвестиции связаны с риском. Всегда консультируйтесь с финансовым советником.
""")