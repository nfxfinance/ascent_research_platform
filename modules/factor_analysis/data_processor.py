import pandas as pd
import numpy as np
import streamlit as st
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理器 - 负责数据验证、准备和处理"""

    @staticmethod
    def validate_and_prepare_data(df: pd.DataFrame, price_col: str, signal_cols: List[str]) -> pd.DataFrame:
        """验证并准备数据用于SignalPerf初始化"""
        # 创建数据副本
        prepared_df = df.copy()

        # 验证必要列是否存在
        missing_cols = []
        if price_col not in prepared_df.columns:
            missing_cols.append(price_col)

        for col in signal_cols:
            if col not in prepared_df.columns:
                missing_cols.append(col)

        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")

        # 检查数据类型
        for col in signal_cols:
            if not pd.api.types.is_numeric_dtype(prepared_df[col]):
                try:
                    prepared_df[col] = pd.to_numeric(prepared_df[col], errors='coerce')
                    logger.warning(f"列 {col} 已转换为数值类型")
                except Exception as e:
                    raise ValueError(f"无法将列 {col} 转换为数值类型: {e}")

        if not pd.api.types.is_numeric_dtype(prepared_df[price_col]):
            try:
                prepared_df[price_col] = pd.to_numeric(prepared_df[price_col], errors='coerce')
                logger.warning(f"价格列 {price_col} 已转换为数值类型")
            except Exception as e:
                raise ValueError(f"无法将价格列 {price_col} 转换为数值类型: {e}")

        # 处理索引
        if 'day' in prepared_df.columns:
            # 如果存在day列，将其设为索引
            prepared_df = prepared_df.set_index('day')
        elif 'date' in prepared_df.columns:
            # 如果存在date列，将其设为索引
            prepared_df = prepared_df.set_index('date')
        elif prepared_df.index.name not in ['day', 'date']:
            # 如果索引不是day或date，重置索引并创建day列
            prepared_df = prepared_df.reset_index()
            if 'index' in prepared_df.columns:
                prepared_df = prepared_df.rename(columns={'index': 'day'})
                prepared_df = prepared_df.set_index('day')
            else:
                prepared_df.index.name = 'day'

        # 确保索引是数值类型
        if not pd.api.types.is_numeric_dtype(prepared_df.index):
            try:
                prepared_df.index = pd.to_numeric(prepared_df.index, errors='coerce')
            except Exception as e:
                # 如果无法转换为数值，创建数值索引
                prepared_df = prepared_df.reset_index(drop=True)
                prepared_df.index.name = 'day'

        # 删除包含NaN的行
        initial_rows = len(prepared_df)
        prepared_df = prepared_df.dropna(subset=[price_col] + signal_cols)
        final_rows = len(prepared_df)

        if initial_rows != final_rows:
            logger.warning(f"删除了 {initial_rows - final_rows} 行包含NaN的数据")

        # 确保有足够的数据
        if len(prepared_df) < 30:
            raise ValueError(f"数据量太少，至少需要30行数据，当前只有{len(prepared_df)}行")

        # 排序数据
        prepared_df = prepared_df.sort_index()

        logger.info(f"数据准备完成: {len(prepared_df)} 行, 价格列: {price_col}, 信号列: {signal_cols}")
        return prepared_df

    @staticmethod
    def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
        """自动检测价格列和信号列"""
        price_col = None
        signal_cols = []

        # 检测价格列
        price_candidates = ['price', 'close', 'close_price', 'adj_close', 'adjusted_close']
        for col in price_candidates:
            if col in df.columns:
                price_col = col
                break

        # 如果没有找到明确的价格列，选择第一个数值列
        if price_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                price_col = numeric_cols[0]

        # 检测信号列（排除价格列和常见的非信号列）
        exclude_cols = ['day', 'date', 'time', 'timestamp',
                        'open', 'high', 'low', 'volume', 'turnover', 'vwap', 'open_price', 'high_price', 'low_price', 'close_price', 'adj_close', 'adjusted_close', 'amount', 'return',
                        'Unnamed: 0', 'asset', 'symbol', 'fsym', 'tsym',
                        price_col]
        if 'day' in df.columns:
            exclude_cols.append('day')
        if 'date' in df.columns:
            exclude_cols.append('date')

        for col in df.columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
                signal_cols.append(col)

        return price_col, signal_cols

    @staticmethod
    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据信息"""
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'non_numeric_columns': df.select_dtypes(exclude=[np.number]).columns.tolist(),
            'index_info': {
                'name': df.index.name,
                'dtype': str(df.index.dtype),
                'range': [df.index.min(), df.index.max()] if len(df) > 0 else [None, None]
            }
        }
        return info

    @staticmethod
    def validate_data_quality(df: pd.DataFrame, price_col: str, signal_cols: List[str]) -> Dict[str, Any]:
        """验证数据质量"""
        quality_report = {
            'total_rows': len(df),
            'missing_data': {},
            'data_types': {},
            'outliers': {},
            'warnings': [],
            'errors': []
        }

        # 检查缺失值
        for col in [price_col] + signal_cols:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                quality_report['missing_data'][col] = {
                    'count': missing_count,
                    'percentage': missing_pct
                }

                if missing_pct > 50:
                    quality_report['errors'].append(f"列 {col} 缺失值过多 ({missing_pct:.1f}%)")
                elif missing_pct > 10:
                    quality_report['warnings'].append(f"列 {col} 缺失值较多 ({missing_pct:.1f}%)")

        # 检查数据类型
        for col in [price_col] + signal_cols:
            if col in df.columns:
                quality_report['data_types'][col] = str(df[col].dtype)
                if not pd.api.types.is_numeric_dtype(df[col]):
                    quality_report['errors'].append(f"列 {col} 不是数值类型")

        # 检查异常值（使用IQR方法）
        for col in signal_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_pct = (outlier_count / len(df)) * 100

                quality_report['outliers'][col] = {
                    'count': outlier_count,
                    'percentage': outlier_pct,
                    'bounds': [lower_bound, upper_bound]
                }

                if outlier_pct > 5:
                    quality_report['warnings'].append(f"列 {col} 异常值较多 ({outlier_pct:.1f}%)")

        # 检查数据量
        if len(df) < 100:
            quality_report['warnings'].append(f"数据量较少 ({len(df)} 行)，分析结果可能不够可靠")

        return quality_report

    @staticmethod
    def clean_data(df: pd.DataFrame, price_col: str, signal_cols: List[str],
                   remove_outliers: bool = False, fill_missing: str = 'drop') -> pd.DataFrame:
        """清洗数据"""
        cleaned_df = df.copy()

        # 处理缺失值
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna(subset=[price_col] + signal_cols)
        elif fill_missing == 'forward':
            cleaned_df[price_col] = cleaned_df[price_col].fillna(method='ffill')
            for col in signal_cols:
                cleaned_df[col] = cleaned_df[col].fillna(method='ffill')
        elif fill_missing == 'backward':
            cleaned_df[price_col] = cleaned_df[price_col].fillna(method='bfill')
            for col in signal_cols:
                cleaned_df[col] = cleaned_df[col].fillna(method='bfill')
        elif fill_missing == 'mean':
            cleaned_df[price_col] = cleaned_df[price_col].fillna(cleaned_df[price_col].mean())
            for col in signal_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())

        # 移除异常值
        if remove_outliers:
            for col in signal_cols:
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    # 将异常值替换为边界值
                    cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)

        return cleaned_df

    @staticmethod
    def create_data_preview(df: pd.DataFrame, max_rows: int = 10) -> pd.DataFrame:
        """创建数据预览"""
        if len(df) <= max_rows:
            return df

        # 显示前几行和后几行
        half_rows = max_rows // 2
        preview_df = pd.concat([
            df.head(half_rows),
            df.tail(max_rows - half_rows)
        ])

        return preview_df
