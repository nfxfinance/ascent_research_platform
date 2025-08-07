#!/usr/bin/env python3

import streamlit as st
import inspect
import ast
import typing
from typing import List, Optional, Dict, Any, Literal

# Type mapping for parameter inference
TYPE_MAPPING = {
    'signal_name': str,
    'signal_names': list,
    'signals': list,
    'lookfwd_day': float,
    'lookfwd_days': list,
    'sample_day': int,
    'sample_days': list,
    'sample_freq_days': int,
    'width': int,
    'height': int,
    'risk_adj': bool,
    'return_window': int,
    'max_lags': int,
    'labeling_method': str,
    'price_col': str,
    'signal_cols': list,
    'day': list,
    'window_size': int,
    'step_size': int,
    'min_periods': int,
    'quantile': float,
    'bins': int,
    'alpha': float,
    'threshold': float,
    'method': str,
    'freq': str,
    'period': int,
    'lag': int,
    'lags': list,
    'n_periods': int,
    'overlap': bool,
}


class ParameterParser:
    """Smart parameter parser for function signatures"""

    @staticmethod
    def infer_parameter_type(name: str, annotation: Any, default_val: Any) -> type:
        """Infer parameter type from name, annotation, and default value"""
        # Use explicit type annotations first
        if annotation and annotation != inspect.Parameter.empty:
            if hasattr(annotation, '__origin__'):
                # Handle generic types like List[int], Optional[str], etc.
                origin = annotation.__origin__
                if origin is list or origin is typing.List:
                    return list
                elif origin is typing.Union:
                    # Union type, get first non-None type
                    for arg in annotation.__args__:
                        if arg != type(None):
                            return arg
            return annotation

        # Use type mapping
        if name in TYPE_MAPPING:
            return TYPE_MAPPING[name]

        # Infer from default value
        if default_val is not None and default_val != inspect.Parameter.empty:
            return type(default_val)

        # Infer from parameter name pattern
        if any(keyword in name.lower() for keyword in ['list', 'names', 'cols', 'days', 'lags', 'periods']):
            return list
        elif any(keyword in name.lower() for keyword in ['adj', 'risk', 'enable', 'show', 'plot']):
            return bool
        elif any(keyword in name.lower() for keyword in ['width', 'height', 'sample', 'window', 'step', 'bins', 'lag', 'period']):
            return int
        elif any(keyword in name.lower() for keyword in ['day', 'rate', 'threshold', 'alpha', 'quantile']):
            return float

        # Default to string
        return str

    @staticmethod
    def get_function_signature(func) -> Dict[str, Any]:
        """Get function signature information"""
        params = inspect.signature(func).parameters
        return {
            name: {
                'annotation': param.annotation,
                'default': param.default if param.default is not inspect.Parameter.empty else None,
                'kind': param.kind
            }
            for name, param in params.items()
            if name not in ['self', 'save_plot', 'save_dir']
        }

    @staticmethod
    def create_form_widget(name: str, param_info: Dict, signal_list: List[str] = None,
                        default_signal: str = None, export_defaults: Dict = None) -> Any:
        """Create form widget based on parameter information"""
        annotation = param_info['annotation']
        default_val = param_info['default']

        # Special handling for signal name
        if name == 'signal_name':
            if signal_list:
                selected_signal = (export_defaults.get(name) if export_defaults else None) or default_signal
                idx = signal_list.index(selected_signal) if selected_signal in signal_list else 0
                return st.selectbox(f"📊 {name}", options=signal_list, index=idx, help="Select the signal to analyze")
            else:
                return st.text_input(f"📊 {name}", value=str(default_val) if default_val else "", help="Input signal name")

        # Special handling for signal name list
        if (name == 'signal_names' or name == 'signals') and signal_list:
            default_selection = (export_defaults.get(name) if export_defaults else None) or signal_list
            return st.multiselect(f"📊 {name}", options=signal_list, default=default_selection, help="Select the signal list to analyze")

        # Handle boolean type
        if annotation is bool:
            return st.checkbox(f"✅ {name}", value=bool(default_val), help=f"Enable {name}")

        # Handle Literal type
        if (hasattr(annotation, '__origin__') and
            (str(annotation.__origin__) == 'typing.Literal' or
             getattr(annotation, '__origin__', None) is getattr(typing, 'Literal', None))):
            options = list(annotation.__args__)
            default_idx = options.index(default_val) if default_val in options else 0
            return st.selectbox(f"🔽 {name}", options=options, index=default_idx, help=f"Select {name} type")

        # Alternative detection for Python < 3.8 using typing_extensions
        if (hasattr(annotation, '__origin__') and
            'Literal' in str(annotation.__origin__)):
            options = list(annotation.__args__)
            default_idx = options.index(default_val) if default_val in options else 0
            return st.selectbox(f"🔽 {name}", options=options, index=default_idx, help=f"Select {name} type")

        # Handle list type (more intelligent handling)
        if ((hasattr(annotation, '__origin__') and annotation.__origin__ in [list, typing.List])
            or isinstance(default_val, list)):
            # Special handling for time-related parameters, adding more detailed help text
            if any(keyword in name.lower() for keyword in ['days', 'day', 'sample', 'lookfwd', 'lookback']):
                help_text = f"Input {name} list, supported formats:\n• List: [1,7,30]\n• Range: range(1,31)\n• Step: range(1,31,2)\n• Comma-separated: 1,7,30"
            else:
                help_text = f"Input {name} list"
            return st.text_input(f"📋 {name}", value=str(default_val), help=help_text)

        # Handle numeric type (more detailed handling)
        if annotation in [int, float] or 'int' in str(annotation) or 'float' in str(annotation):
            try:
                if 'int' in str(annotation) or annotation is int:
                    default_num = int(default_val) if default_val is not None else 0
                    # Set reasonable range and step for time-related parameters
                    if any(keyword in name.lower() for keyword in ['day', 'days', 'sample', 'lookfwd', 'lookback']):
                        return st.number_input(f"🔢 {name}", value=default_num, min_value=0, max_value=1000, step=1, help=f"Set {name} value (days)")
                    elif any(keyword in name.lower() for keyword in ['width', 'height']):
                        return st.number_input(f"🔢 {name}", value=default_num, min_value=0, max_value=4000, step=50, help=f"Set {name} value (chart size)")
                    else:
                        return st.number_input(f"🔢 {name}", value=default_num, help=f"Set {name} value")
                else:
                    default_num = float(default_val) if default_val is not None else 0.0
                    if any(keyword in name.lower() for keyword in ['ratio', 'rate', 'alpha', 'confidence', 'threshold']):
                        return st.number_input(f"🔢 {name}", value=default_num, min_value=0.0, max_value=1.0, step=0.01, help=f"Set {name} value (ratio)")
                    else:
                        return st.number_input(f"🔢 {name}", value=default_num, help=f"Set {name} value")
            except (TypeError, ValueError):
                return st.text_input(f"📝 {name}", value=str(default_val) if default_val else "", help=f"Input {name} value")

        # Handle special cases for string type
        if annotation is str or (hasattr(annotation, '__origin__') and annotation.__origin__ is str):
            # If the parameter name indicates an expression (e.g., range expression)
            if 'range' in str(default_val) or 'np.' in str(default_val):
                return st.text_input(f"📝 {name}", value=str(default_val), help=f"Input {name} expression, e.g., range(1, 31)")
            else:
                return st.text_input(f"📝 {name}", value=str(default_val) if default_val else "", help=f"Input {name} value")

        # Handle special cases for Union type
        if hasattr(annotation, '__origin__') and annotation.__origin__ is typing.Union:
            # Extract Union types
            union_types = annotation.__args__
            type_names = [getattr(t, '__name__', str(t)) for t in union_types if t != type(None)]

            if len(type_names) == 1:
                # If it's just an Optional type, handle it as a single type
                return ParameterParser.create_form_widget(name, {'annotation': union_types[0], 'default': default_val}, signal_list, default_signal, export_defaults)
            else:
                # Multiple Union types, use text input
                return st.text_input(f"📝 {name}", value=str(default_val) if default_val else "", help=f"Input {name} value (supported types: {', '.join(type_names)})")

        # Default to text input
        return st.text_input(f"📝 {name}", value=str(default_val) if default_val else "", help=f"Input {name} value")

    @staticmethod
    def create_compact_parameter_form(signature: Dict[str, Any], signal_list: List[str] = None,
                                    default_signal: str = None, export_defaults: Dict = None) -> Dict[str, Any]:
        """Create compact parameter form with intelligent grouping and multi-column display"""
        # Merge function default values and export_defaults
        merged_signature = {}
        for name, param_info in signature.items():
            merged_param_info = param_info.copy()
            if export_defaults and name in export_defaults:
                merged_param_info['default'] = export_defaults[name]
            merged_signature[name] = merged_param_info

        # Parameter grouping
        param_groups = {
            'primary': [],      # Primary parameters: signal related
            'time': [],         # Time parameters: day, days, window, etc.
            'display': [],      # Display parameters: width, height
            'numeric': [],      # Numeric parameters: threshold, alpha, etc.
            'boolean': [],      # Boolean parameters
            'advanced': []      # Advanced parameters: others
        }

        for name, param_info in merged_signature.items():
            annotation = param_info['annotation']

            if 'signal' in name.lower():
                param_groups['primary'].append((name, param_info))
            elif any(keyword in name.lower() for keyword in ['day', 'days', 'window', 'period', 'lag', 'sample', 'freq']):
                param_groups['time'].append((name, param_info))
            elif any(keyword in name.lower() for keyword in ['width', 'height']):
                param_groups['display'].append((name, param_info))
            elif annotation is bool:
                param_groups['boolean'].append((name, param_info))
            elif any(keyword in name.lower() for keyword in ['threshold', 'alpha', 'confidence', 'ratio', 'rate']):
                param_groups['numeric'].append((name, param_info))
            else:
                param_groups['advanced'].append((name, param_info))

        form_data = {}

        # Render primary parameters (single column)
        if param_groups['primary']:
            # st.markdown("**📊 Signal Parameters**")
            for name, param_info in param_groups['primary']:
                form_data[name] = ParameterParser.create_form_widget(name, param_info, signal_list, default_signal, export_defaults)

        # Render time parameters (2 columns)
        if param_groups['time']:
            # st.markdown("**⏰ Time Parameters**")
            time_params = param_groups['time']
            for i in range(0, len(time_params), 6):
                cols = st.columns(6)
                for j, col in enumerate(cols):
                    if i + j < len(time_params):
                        name, param_info = time_params[i + j]
                        with col:
                            form_data[name] = ParameterParser.create_form_widget(name, param_info, signal_list, default_signal, export_defaults)

        # Render display parameters (2 columns)
        if param_groups['display']:
            # st.markdown("**🎨 Display Parameters**")
            display_params = param_groups['display']
            for i in range(0, len(display_params), 6):
                cols = st.columns(6)
                for j, col in enumerate(cols):
                    if i + j < len(display_params):
                        name, param_info = display_params[i + j]
                        with col:
                            form_data[name] = ParameterParser.create_form_widget(name, param_info, signal_list, default_signal, export_defaults)

        # Render numeric parameters (3 columns)
        if param_groups['numeric']:
            # st.markdown("**🔢 Numeric Parameters**")
            numeric_params = param_groups['numeric']
            for i in range(0, len(numeric_params), 6):
                cols = st.columns(6)
                for j, col in enumerate(cols):
                    if i + j < len(numeric_params):
                        name, param_info = numeric_params[i + j]
                        with col:
                            form_data[name] = ParameterParser.create_form_widget(name, param_info, signal_list, default_signal, export_defaults)

        # Render boolean parameters (4 columns)
        if param_groups['boolean']:
            # st.markdown("**✅ Boolean Parameters**")
            boolean_params = param_groups['boolean']
            for i in range(0, len(boolean_params), 6):
                cols = st.columns(6)
                for j, col in enumerate(cols):
                    if i + j < len(boolean_params):
                        name, param_info = boolean_params[i + j]
                        with col:
                            form_data[name] = ParameterParser.create_form_widget(name, param_info, signal_list, default_signal, export_defaults)

        # Render advanced parameters (2 columns)
        if param_groups['advanced']:
            # with st.expander("🔧 Advanced Parameters", expanded=False):
            # st.markdown("**🔧 Advanced Parameters**")
            advanced_params = param_groups['advanced']
            for i in range(0, len(advanced_params), 6):
                cols = st.columns(6)
                for j, col in enumerate(cols):
                    if i + j < len(advanced_params):
                        name, param_info = advanced_params[i + j]
                        with col:
                            form_data[name] = ParameterParser.create_form_widget(name, param_info, signal_list, default_signal, export_defaults)

        return form_data

    @staticmethod
    def process_form_data(form_data: Dict[str, Any], func_signature: Dict[str, Any]) -> Dict[str, Any]:
        """Process form data, convert to correct type"""
        processed = {}

        for name, value in form_data.items():
            param_info = func_signature.get(name, {})
            annotation = param_info.get('annotation', inspect.Parameter.empty)
            default_val = param_info.get('default', inspect.Parameter.empty)

            # Infer parameter type
            inferred_type = ParameterParser.infer_parameter_type(name, annotation, default_val)

            try:
                # Handle Literal type first - values from selectbox are already the correct type
                if (hasattr(annotation, '__origin__') and
                    (str(annotation.__origin__) == 'typing.Literal' or
                     getattr(annotation, '__origin__', None) is getattr(typing, 'Literal', None) or
                     'Literal' in str(annotation.__origin__))):
                    # For Literal types, the value from selectbox is already the correct type
                    processed[name] = value
                    continue

                # If value is already correct type (from presets), use directly
                if not isinstance(value, str):
                    processed[name] = value
                    continue

                # Handle list type
                if inferred_type is list or (isinstance(default_val, list) and default_val != inspect.Parameter.empty):
                    if isinstance(value, str):
                        # Try safe evaluation
                        if value.strip().startswith('[') and value.strip().endswith(']'):
                            processed[name] = ast.literal_eval(value)
                        elif 'range(' in value:
                            # Handle range expressions
                            safe_globals = {"__builtins__": {}}
                            safe_locals = {"range": range}
                            try:
                                result = eval(value, safe_globals, safe_locals)
                                processed[name] = list(result) if hasattr(result, '__iter__') else [result]
                            except:
                                st.error(f"❌ Can't parse {name}'s range expression: {value}")
                                processed[name] = default_val if default_val != inspect.Parameter.empty else []
                        else:
                            # Try parsing comma-separated values
                            try:
                                processed[name] = [float(x.strip()) if '.' in x.strip() else int(x.strip())
                                                 for x in value.split(',')]
                            except:
                                processed[name] = ast.literal_eval(value)
                    else:
                        processed[name] = value

                # Handle numeric types
                elif inferred_type in [int, float]:
                    if inferred_type is int:
                        processed[name] = int(float(value))  # Convert to float first then int, handle "1.0" cases
                    else:
                        processed[name] = float(value)

                # Handle boolean type
                elif inferred_type is bool:
                    if isinstance(value, bool):
                        processed[name] = value
                    else:
                        processed[name] = str(value).lower() in ['true', '1', 'yes', 'on']

                # Handle string containing range expressions
                elif isinstance(value, str) and ('range(' in value or 'np.' in value):
                    safe_globals = {"__builtins__": {}}
                    safe_locals = {"range": range, "list": list}
                    try:
                        result = eval(value, safe_globals, safe_locals)
                        processed[name] = list(result) if hasattr(result, '__iter__') and not isinstance(result, str) else result
                    except Exception as e:
                        st.error(f"❌ Can't parse {name}'s expression: {value}. Error: {e}")
                        processed[name] = default_val if default_val != inspect.Parameter.empty else value
                elif isinstance(value, str):
                    if value is not None and value != '':
                        processed[name] = value

                else:
                    processed[name] = value

            except (ValueError, SyntaxError) as e:
                st.error(f"❌ Parameter {name} format error: {e}")
                processed[name] = default_val if default_val != inspect.Parameter.empty else value

        return processed
