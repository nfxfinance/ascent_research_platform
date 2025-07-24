#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import io
import logging
import traceback
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)


def create_download_mhtml_button():
    """Create a button to download current page as MHTML"""
    download_js = """
    <script>
    function downloadMHTML() {
        // Trigger browser save dialog
        const event = new KeyboardEvent('keydown', {
            key: 's',
            code: 'KeyS',
            ctrlKey: true,
            bubbles: true,
            cancelable: true
        });
        document.dispatchEvent(event);
    }
    </script>

    <div style="text-align: center; margin: 20px 0;">
        <button onclick="downloadMHTML()"
                style="
                    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 16px;
                    font-weight: bold;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    transition: transform 0.2s;
                "
                onmouseover="this.style.transform='scale(1.05)'"
                onmouseout="this.style.transform='scale(1)'">
            📄 Save Page as MHTML
        </button>
    </div>
    """
    components.html(download_js, height=100)


class ChartRenderer:
    """Chart renderer for different types of visualizations"""

    @staticmethod
    def render_plotly_chart(chart, key: str = None):
        """Render Plotly chart"""
        try:
            # Validate chart is a valid Plotly chart
            if not hasattr(chart, 'to_html') or not callable(getattr(chart, 'to_html')):
                st.error("❌ Invalid Plotly chart object")
                return

            # Ensure chart has data
            if not hasattr(chart, 'data') or len(chart.data) == 0:
                st.warning("⚠️ Plotly chart has no data")
                return

            # Use streamlit's plotly_chart for rendering
            st.plotly_chart(chart, use_container_width=False, key=key)
            logger.info(f"Plotly chart rendered successfully (key: {key})")

        except Exception as e:
            st.error(f"❌ Plotly chart render failed: {e}")
            logger.error(f"Plotly chart render error: {e}\n{traceback.format_exc()}")

            # Show debug information
            with st.expander("🔍 Plotly chart debug information", expanded=False):
                st.write(f"Chart type: {type(chart)}")
                if hasattr(chart, 'data'):
                    st.write(f"Data trace count: {len(chart.data)}")
                if hasattr(chart, 'layout'):
                    st.write(f"Layout title: {getattr(chart.layout, 'title', 'N/A')}")

    @staticmethod
    def render_matplotlib_chart(fig, key=None):
        """Render Matplotlib chart"""
        try:
            if hasattr(fig, '_closed') and fig._closed:
                st.warning("⚠️ Matplotlib chart is closed, cannot display")
                return

            # Check if it's a valid matplotlib figure
            if not hasattr(fig, 'savefig'):
                st.error("⚠️ Invalid Matplotlib chart object")
                return

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.image(buf, use_container_width=True)
            buf.close()

        except Exception as e:
            st.error(f"Render Matplotlib chart error: {e}")
            logger.error(f"Matplotlib chart render error: {e}\n{traceback.format_exc()}")

    @staticmethod
    def render_dataframe(df, key=None):
        """Render DataFrame"""
        if isinstance(df, pd.DataFrame):
            df_display = df.copy()
            numeric_cols = df_display.select_dtypes(include=[np.number]).columns
            df_display[numeric_cols] = df_display[numeric_cols].round(4)
            st.dataframe(df_display, use_container_width=True, height=400, key=key)
        else:
            st.text(str(df))

    @staticmethod
    def render_analysis_output(result, key: str = None):
        """Smartly render analysis results"""
        try:
            # Check if result is None
            if result is None:
                st.warning("⚠️ Analysis returned no results")
                return

            # If result is a tuple, take the first element
            if isinstance(result, tuple):
                if len(result) > 0:
                    result = result[0]
                else:
                    st.warning("⚠️ Analysis returned empty tuple")
                    return

            # Detect plotly chart - more strict checking
            if hasattr(result, '__module__') and result.__module__ and 'plotly' in result.__module__:
                # Further validate if it's a valid plotly chart
                if hasattr(result, 'data') and hasattr(result, 'layout'):
                    ChartRenderer.render_plotly_chart(result, key=key)
                    return
                else:
                    st.warning("⚠️ Detected Plotly object but data structure is incomplete")
                    logger.warning(f"Invalid plotly object structure: {type(result)}")

            # Detect matplotlib chart - more strict checking
            if hasattr(result, 'savefig') and callable(getattr(result, 'savefig')):
                # Validate if it's a valid matplotlib chart
                if hasattr(result, 'axes') or hasattr(result, 'get_axes'):
                    ChartRenderer.render_matplotlib_chart(result, key=key)
                    return
                else:
                    st.warning("⚠️ Detected Matplotlib object but structure is incomplete")

            # Detect pandas DataFrame
            if isinstance(result, pd.DataFrame):
                if not result.empty:
                    ChartRenderer.render_dataframe(result, key=key)
                    return
                else:
                    st.warning("⚠️ Analysis returned empty DataFrame")
                    return

            # Detect string or other types
            if isinstance(result, str):
                if result.strip():  # Non-empty string
                    st.text(result)
                else:
                    st.warning("⚠️ Analysis returned empty string")
                return

            # Detect numeric types
            if isinstance(result, (int, float, complex)):
                st.metric("Analysis result", result)
                return

            # Detect list or array
            if isinstance(result, (list, tuple, np.ndarray)):
                if len(result) > 0:
                    # Try to convert to more friendly display format
                    if isinstance(result, np.ndarray):
                        if result.ndim == 1 and len(result) <= 10:
                            # Small 1D array displayed as table
                            df_result = pd.DataFrame({'Value': result})
                            st.dataframe(df_result, use_container_width=True)
                        else:
                            st.write(result)
                    else:
                        st.write(result)
                else:
                    st.warning("⚠️ Analysis returned empty list/array")
                return

            # Detect dictionary
            if isinstance(result, dict):
                if result:  # Non-empty dictionary
                    # Try to convert to better display format
                    try:
                        df_result = pd.DataFrame.from_dict(result, orient='index', columns=['Value'])
                        st.dataframe(df_result, use_container_width=True)
                    except:
                        st.json(result)
                else:
                    st.warning("⚠️ Analysis returned empty dictionary")
                return

            # Other types, try to convert to string for display
            result_str = str(result)
            if result_str and result_str != 'None':
                # Check if it contains useful information
                if len(result_str) > 10 or any(char.isalnum() for char in result_str):
                    st.text(result_str)
                else:
                    st.warning("⚠️ Analysis returned meaningless result")
                    with st.expander("🔍 Original result", expanded=False):
                        st.write(f"Type: {type(result)}")
                        st.write(f"Content: {repr(result)}")
            else:
                st.warning("⚠️ Analysis returned result type that cannot be displayed")
                with st.expander("🔍 Debug information", expanded=False):
                    st.write(f"Result type: {type(result)}")
                    st.write(f"Result content: {repr(result)}")

        except Exception as e:
            st.error(f"❌ Error rendering analysis results: {e}")
            logger.error(f"Analysis output render error: {e}\n{traceback.format_exc()}")

            # Show detailed debug information
            with st.expander("🔍 Detailed error information", expanded=False):
                st.code(traceback.format_exc())
                st.write("**Result information:**")
                try:
                    st.write(f"Type: {type(result)}")
                    st.write(f"Content: {repr(result)[:1000]}...")  # Limit display length
                except:
                    st.write("Cannot display result content")

    @staticmethod
    def render_side_by_side_charts(chart1, chart2, chart1_title="Chart 1", chart2_title="Chart 2", key=None):
        """Render two charts side by side with controlled size"""
        try:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**{chart1_title}**")
                if hasattr(chart1, 'savefig') and callable(getattr(chart1, 'savefig')):
                    # Matplotlib chart
                    buf1 = io.BytesIO()
                    chart1.savefig(buf1, format='png', dpi=150, bbox_inches='tight')
                    buf1.seek(0)
                    st.image(buf1, use_container_width=True)
                    buf1.close()
                elif hasattr(chart1, '__module__') and chart1.__module__ and 'plotly' in chart1.__module__:
                    # Plotly chart
                    st.plotly_chart(chart1, use_container_width=False)
                else:
                    st.warning("⚠️ Chart 1 type not supported")

            with col2:
                st.markdown(f"**{chart2_title}**")
                if hasattr(chart2, 'savefig') and callable(getattr(chart2, 'savefig')):
                    # Matplotlib chart
                    buf2 = io.BytesIO()
                    chart2.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
                    buf2.seek(0)
                    st.image(buf2, use_container_width=True)
                    buf2.close()
                elif hasattr(chart2, '__module__') and chart2.__module__ and 'plotly' in chart2.__module__:
                    # Plotly chart
                    st.plotly_chart(chart2, use_container_width=False)
                else:
                    st.warning("⚠️ Chart 2 type not supported")

        except Exception as e:
            st.error(f"❌ Side-by-side chart render failed: {e}")
            logger.error(f"Side-by-side chart render error: {e}\n{traceback.format_exc()}")
