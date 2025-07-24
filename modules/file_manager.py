#!/usr/bin/env python3
"""
文件管理模块 - 实现文件上传到服务器和数据库记录管理
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import sqlite3
import hashlib
import shutil
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
import mimetypes
from io import BytesIO

logger = logging.getLogger(__name__)

class FileManager:
    """文件管理器 - 处理文件上传和数据库记录"""

    def __init__(self, upload_dir: str = "uploads", db_path: str = "data/file_records.db"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.initialize_database()

    def initialize_database(self):
        """初始化数据库表结构"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 创建文件记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_type TEXT NOT NULL,
                    mime_type TEXT,
                    file_hash TEXT UNIQUE NOT NULL,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    uploaded_by TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    download_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP,
                    metadata TEXT
                )
            ''')

            # 创建文件分类表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category_name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 创建文件标签表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tag_name TEXT UNIQUE NOT NULL,
                    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 插入默认分类
            default_categories = [
                ('数据文件', '包含CSV、Excel等数据文件'),
                ('文档', '包含PDF、Word等文档文件'),
                ('图片', '包含图片文件'),
                ('代码', '包含Python、SQL等代码文件'),
                ('其他', '其他类型文件')
            ]

            cursor.executemany('''
                INSERT OR IGNORE INTO file_categories (category_name, description)
                VALUES (?, ?)
            ''', default_categories)

            conn.commit()
            conn.close()
            logger.info("数据库初始化完成")

        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise

    def calculate_file_hash(self, file_content: bytes) -> str:
        """计算文件哈希值"""
        return hashlib.sha256(file_content).hexdigest()

    def get_file_type_category(self, filename: str) -> str:
        """根据文件扩展名获取文件类型分类"""
        ext = Path(filename).suffix.lower()

        data_extensions = {'.csv', '.xlsx', '.xls', '.json', '.parquet', '.pkl', '.pickle'}
        doc_extensions = {'.pdf', '.doc', '.docx', '.txt', '.md', '.rtf'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'}
        code_extensions = {'.py', '.sql', '.js', '.html', '.css', '.r', '.ipynb'}

        if ext in data_extensions:
            return '数据文件'
        elif ext in doc_extensions:
            return '文档'
        elif ext in image_extensions:
            return '图片'
        elif ext in code_extensions:
            return '代码'
        else:
            return '其他'

    def upload_file(self, uploaded_file, description: str = "", tags: str = "",
                   username: str = "unknown") -> Tuple[bool, str, Optional[int]]:
        """
        上传文件到服务器并记录到数据库

        Returns:
            Tuple[bool, str, Optional[int]]: (成功标志, 消息, 文件记录ID)
        """
        try:
            # 读取文件内容
            file_content = uploaded_file.getvalue()
            file_size = len(file_content)

            # 计算文件哈希
            file_hash = self.calculate_file_hash(file_content)

            # 检查文件是否已存在
            if self.check_file_exists(file_hash):
                return False, "文件已存在，请勿重复上传", None

            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = Path(uploaded_file.name).suffix
            unique_filename = f"{timestamp}_{file_hash[:8]}{file_extension}"

            # 保存文件到服务器
            file_path = self.upload_dir / unique_filename
            with open(file_path, 'wb') as f:
                f.write(file_content)

            # 获取文件MIME类型
            mime_type, _ = mimetypes.guess_type(uploaded_file.name)

            # 获取文件类型分类
            file_type = self.get_file_type_category(uploaded_file.name)

            # 准备元数据
            metadata = {
                'upload_timestamp': datetime.now().isoformat(),
                'file_extension': file_extension,
                'original_size': file_size
            }

            # 记录到数据库
            record_id = self.save_file_record(
                filename=unique_filename,
                original_filename=uploaded_file.name,
                file_path=str(file_path),
                file_size=file_size,
                file_type=file_type,
                mime_type=mime_type,
                file_hash=file_hash,
                uploaded_by=username,
                description=description,
                tags=tags,
                metadata=json.dumps(metadata)
            )

            if record_id:
                return True, f"文件上传成功！文件ID: {record_id}", record_id
            else:
                # 如果数据库记录失败，删除已上传的文件
                file_path.unlink(missing_ok=True)
                return False, "数据库记录失败，文件上传失败", None

        except Exception as e:
            logger.error(f"文件上传失败: {e}")
            return False, f"文件上传失败: {str(e)}", None

    def check_file_exists(self, file_hash: str) -> bool:
        """检查文件是否已存在"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id FROM file_records
                WHERE file_hash = ? AND is_active = 1
            ''', (file_hash,))

            result = cursor.fetchone()
            conn.close()

            return result is not None

        except Exception as e:
            logger.error(f"检查文件是否存在失败: {e}")
            return False

    def save_file_record(self, filename: str, original_filename: str, file_path: str,
                        file_size: int, file_type: str, mime_type: str, file_hash: str,
                        uploaded_by: str, description: str = "", tags: str = "",
                        metadata: str = "") -> Optional[int]:
        """保存文件记录到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO file_records (
                    filename, original_filename, file_path, file_size, file_type,
                    mime_type, file_hash, uploaded_by, description, tags, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (filename, original_filename, file_path, file_size, file_type,
                  mime_type, file_hash, uploaded_by, description, tags, metadata))

            record_id = cursor.lastrowid
            conn.commit()
            conn.close()

            logger.info(f"文件记录保存成功，ID: {record_id}")
            return record_id

        except Exception as e:
            logger.error(f"保存文件记录失败: {e}")
            return None

    def get_file_records(self, limit: int = 100, offset: int = 0,
                        file_type: str = None, search_term: str = None) -> List[Dict]:
        """获取文件记录列表"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 构建查询条件
            where_conditions = ["is_active = 1"]
            params = []

            if file_type and file_type != "全部":
                where_conditions.append("file_type = ?")
                params.append(file_type)

            if search_term:
                where_conditions.append(
                    "(original_filename LIKE ? OR description LIKE ? OR tags LIKE ?)"
                )
                search_pattern = f"%{search_term}%"
                params.extend([search_pattern, search_pattern, search_pattern])

            where_clause = " AND ".join(where_conditions)
            params.extend([limit, offset])

            cursor.execute(f'''
                SELECT id, filename, original_filename, file_path, file_size, file_type,
                       mime_type, file_hash, upload_time, uploaded_by, description, tags,
                       download_count, last_accessed, metadata
                FROM file_records
                WHERE {where_clause}
                ORDER BY upload_time DESC
                LIMIT ? OFFSET ?
            ''', params)

            records = []
            for row in cursor.fetchall():
                records.append({
                    'id': row[0],
                    'filename': row[1],
                    'original_filename': row[2],
                    'file_path': row[3],
                    'file_size': row[4],
                    'file_type': row[5],
                    'mime_type': row[6],
                    'file_hash': row[7],
                    'upload_time': row[8],
                    'uploaded_by': row[9],
                    'description': row[10],
                    'tags': row[11],
                    'download_count': row[12],
                    'last_accessed': row[13],
                    'metadata': row[14]
                })

            conn.close()
            return records

        except Exception as e:
            logger.error(f"获取文件记录失败: {e}")
            return []

    def get_file_statistics(self) -> Dict[str, Any]:
        """获取文件统计信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 总文件数
            cursor.execute("SELECT COUNT(*) FROM file_records WHERE is_active = 1")
            total_files = cursor.fetchone()[0]

            # 总文件大小
            cursor.execute("SELECT SUM(file_size) FROM file_records WHERE is_active = 1")
            total_size = cursor.fetchone()[0] or 0

            # 按类型统计
            cursor.execute('''
                SELECT file_type, COUNT(*), SUM(file_size)
                FROM file_records
                WHERE is_active = 1
                GROUP BY file_type
            ''')
            type_stats = cursor.fetchall()

            # 最近上传的文件
            cursor.execute('''
                SELECT original_filename, upload_time, uploaded_by
                FROM file_records
                WHERE is_active = 1
                ORDER BY upload_time DESC
                LIMIT 5
            ''')
            recent_files = cursor.fetchall()

            conn.close()

            return {
                'total_files': total_files,
                'total_size': total_size,
                'type_stats': type_stats,
                'recent_files': recent_files
            }

        except Exception as e:
            logger.error(f"获取文件统计信息失败: {e}")
            return {
                'total_files': 0,
                'total_size': 0,
                'type_stats': [],
                'recent_files': []
            }

    def delete_file(self, file_id: int) -> Tuple[bool, str]:
        """删除文件（软删除）"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 获取文件信息
            cursor.execute('''
                SELECT file_path FROM file_records
                WHERE id = ? AND is_active = 1
            ''', (file_id,))

            result = cursor.fetchone()
            if not result:
                conn.close()
                return False, "文件不存在或已删除"

            file_path = Path(result[0])

            # 软删除数据库记录
            cursor.execute('''
                UPDATE file_records
                SET is_active = 0
                WHERE id = ?
            ''', (file_id,))

            # 删除物理文件
            if file_path.exists():
                file_path.unlink()

            conn.commit()
            conn.close()

            logger.info(f"文件删除成功，ID: {file_id}")
            return True, "文件删除成功"

        except Exception as e:
            logger.error(f"删除文件失败: {e}")
            return False, f"删除文件失败: {str(e)}"

    def download_file(self, file_id: int) -> Tuple[bool, str, Optional[bytes], Optional[str]]:
        """下载文件"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 获取文件信息
            cursor.execute('''
                SELECT file_path, original_filename, mime_type
                FROM file_records
                WHERE id = ? AND is_active = 1
            ''', (file_id,))

            result = cursor.fetchone()
            if not result:
                conn.close()
                return False, "文件不存在或已删除", None, None

            file_path, original_filename, mime_type = result

            # 读取文件内容
            file_path = Path(file_path)
            if not file_path.exists():
                conn.close()
                return False, "文件在服务器上不存在", None, None

            with open(file_path, 'rb') as f:
                file_content = f.read()

            # 更新下载计数和最后访问时间
            cursor.execute('''
                UPDATE file_records
                SET download_count = download_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (file_id,))

            conn.commit()
            conn.close()

            return True, "文件下载成功", file_content, original_filename

        except Exception as e:
            logger.error(f"下载文件失败: {e}")
            return False, f"下载文件失败: {str(e)}", None, None

    def update_file_info(self, file_id: int, description: str = None,
                        tags: str = None) -> Tuple[bool, str]:
        """更新文件信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 构建更新语句
            update_fields = []
            params = []

            if description is not None:
                update_fields.append("description = ?")
                params.append(description)

            if tags is not None:
                update_fields.append("tags = ?")
                params.append(tags)

            if not update_fields:
                conn.close()
                return False, "没有需要更新的字段"

            params.append(file_id)

            cursor.execute(f'''
                UPDATE file_records
                SET {", ".join(update_fields)}
                WHERE id = ? AND is_active = 1
            ''', params)

            if cursor.rowcount == 0:
                conn.close()
                return False, "文件不存在或已删除"

            conn.commit()
            conn.close()

            return True, "文件信息更新成功"

        except Exception as e:
            logger.error(f"更新文件信息失败: {e}")
            return False, f"更新文件信息失败: {str(e)}"

    def format_file_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1

        return f"{size_bytes:.1f} {size_names[i]}"

class FileManagerModule:
    """文件管理模块 - Streamlit界面"""

    def __init__(self):
        self.name = "File Manager"
        self.description = "文件上传、管理和数据库记录"
        self.file_manager = FileManager()
        self.initialize_state()

    def initialize_state(self):
        """初始化会话状态"""
        if 'file_manager_current_page' not in st.session_state:
            st.session_state.file_manager_current_page = 0
        if 'file_manager_search_term' not in st.session_state:
            st.session_state.file_manager_search_term = ""
        if 'file_manager_filter_type' not in st.session_state:
            st.session_state.file_manager_filter_type = "全部"

    def render(self):
        """渲染文件管理模块界面"""
        st.markdown("## 📁 文件管理模块")
        st.markdown("*文件上传、管理和数据库记录系统*")

        # 主功能标签页
        tab1, tab2, tab3, tab4 = st.tabs([
            "📤 文件上传",
            "📋 文件列表",
            "📊 统计信息",
            "⚙️ 系统设置"
        ])

        with tab1:
            self.render_file_upload()

        with tab2:
            self.render_file_list()

        with tab3:
            self.render_statistics()

        with tab4:
            self.render_settings()

    def render_file_upload(self):
        """渲染文件上传界面"""
        st.markdown("### 📤 文件上传")

        with st.form("file_upload_form"):
            col1, col2 = st.columns([2, 1])

            with col1:
                uploaded_files = st.file_uploader(
                    "选择要上传的文件",
                    accept_multiple_files=True,
                    help="支持多文件上传，自动检测文件类型"
                )

            with col2:
                st.markdown("**支持的文件类型:**")
                st.markdown("""
                - 📊 数据文件: CSV, Excel, JSON, Parquet
                - 📄 文档: PDF, Word, TXT, Markdown
                - 🖼️ 图片: JPG, PNG, GIF, SVG
                - 💻 代码: Python, SQL, HTML, CSS
                - 📦 其他: 所有其他格式
                """)

            st.markdown("---")

            # 文件描述和标签
            col1, col2 = st.columns(2)

            with col1:
                description = st.text_area(
                    "文件描述",
                    placeholder="请输入文件描述信息...",
                    height=100
                )

            with col2:
                tags = st.text_input(
                    "标签",
                    placeholder="请输入标签，用逗号分隔",
                    help="例如: 数据分析,股票,量化"
                )

            # 上传按钮
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submit_button = st.form_submit_button(
                    "🚀 开始上传",
                    use_container_width=True
                )

        # 处理文件上传
        if submit_button and uploaded_files:
            username = st.session_state.username

            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()

            success_count = 0
            total_files = len(uploaded_files)

            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"正在上传: {uploaded_file.name}")

                success, message, file_id = self.file_manager.upload_file(
                    uploaded_file, description, tags, username
                )

                if success:
                    success_count += 1
                    st.success(f"✅ {uploaded_file.name}: {message}")
                else:
                    st.error(f"❌ {uploaded_file.name}: {message}")

                # 更新进度条
                progress_bar.progress((i + 1) / total_files)

            status_text.text(f"上传完成！成功: {success_count}/{total_files}")

            if success_count > 0:
                st.balloons()
                st.info(f"🎉 成功上传 {success_count} 个文件！")

    def render_file_list(self):
        """渲染文件列表界面"""
        st.markdown("### 📋 文件列表")

        # 搜索和过滤
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            search_term = st.text_input(
                "🔍 搜索文件",
                value=st.session_state.file_manager_search_term,
                placeholder="输入文件名、描述或标签进行搜索..."
            )
            if search_term != st.session_state.file_manager_search_term:
                st.session_state.file_manager_search_term = search_term
                st.session_state.file_manager_current_page = 0

        with col2:
            filter_type = st.selectbox(
                "📁 文件类型",
                ["全部", "数据文件", "文档", "图片", "代码", "其他"],
                index=["全部", "数据文件", "文档", "图片", "代码", "其他"].index(
                    st.session_state.file_manager_filter_type
                )
            )
            if filter_type != st.session_state.file_manager_filter_type:
                st.session_state.file_manager_filter_type = filter_type
                st.session_state.file_manager_current_page = 0

        with col3:
            if st.button("🔄 刷新列表"):
                st.rerun()

        # 获取文件列表
        page_size = 10
        offset = st.session_state.file_manager_current_page * page_size

        file_records = self.file_manager.get_file_records(
            limit=page_size,
            offset=offset,
            file_type=filter_type if filter_type != "全部" else None,
            search_term=search_term if search_term else None
        )

        if not file_records:
            st.info("📭 没有找到符合条件的文件")
            return

        # 显示文件列表
        for record in file_records:
            with st.expander(
                f"📄 {record['original_filename']} "
                f"({self.file_manager.format_file_size(record['file_size'])})"
            ):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**文件ID:** {record['id']}")
                    st.markdown(f"**文件类型:** {record['file_type']}")
                    st.markdown(f"**上传时间:** {record['upload_time']}")
                    st.markdown(f"**上传者:** {record['uploaded_by']}")
                    st.markdown(f"**下载次数:** {record['download_count']}")

                    if record['description']:
                        st.markdown(f"**描述:** {record['description']}")

                    if record['tags']:
                        st.markdown(f"**标签:** {record['tags']}")

                with col2:
                    # 操作按钮
                    if st.button(f"📥 下载", key=f"download_{record['id']}"):
                        self.download_file_action(record['id'])

                    if st.button(f"✏️ 编辑", key=f"edit_{record['id']}"):
                        self.edit_file_action(record['id'])

                    if st.button(f"🗑️ 删除", key=f"delete_{record['id']}"):
                        self.delete_file_action(record['id'])

                    # 加载到数据管理
                    if record['file_type'] == '数据文件':
                        if st.button(f"📊 加载数据", key=f"load_{record['id']}"):
                            self.load_to_data_management(record['id'])

        # 分页控制
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

        with col1:
            if st.button("⏮️ 首页") and st.session_state.file_manager_current_page > 0:
                st.session_state.file_manager_current_page = 0
                st.rerun()

        with col2:
            if st.button("◀️ 上一页") and st.session_state.file_manager_current_page > 0:
                st.session_state.file_manager_current_page -= 1
                st.rerun()

        with col3:
            st.markdown(f"**第 {st.session_state.file_manager_current_page + 1} 页**")

        with col4:
            if st.button("▶️ 下一页") and len(file_records) == page_size:
                st.session_state.file_manager_current_page += 1
                st.rerun()

        with col5:
            if st.button("⏭️ 末页") and len(file_records) == page_size:
                # 这里可以计算总页数，暂时简化处理
                st.session_state.file_manager_current_page += 10
                st.rerun()

    def render_statistics(self):
        """渲染统计信息界面"""
        st.markdown("### 📊 统计信息")

        # 获取统计数据
        stats = self.file_manager.get_file_statistics()

        # 总体统计
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "总文件数",
                stats['total_files'],
                help="系统中所有活跃文件的数量"
            )

        with col2:
            st.metric(
                "总存储空间",
                self.file_manager.format_file_size(stats['total_size']),
                help="所有文件占用的总存储空间"
            )

        with col3:
            avg_size = stats['total_size'] / stats['total_files'] if stats['total_files'] > 0 else 0
            st.metric(
                "平均文件大小",
                self.file_manager.format_file_size(int(avg_size)),
                help="所有文件的平均大小"
            )

        with col4:
            st.metric(
                "文件类型数",
                len(stats['type_stats']),
                help="系统中不同文件类型的数量"
            )

        st.markdown("---")

        # 文件类型分布
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 文件类型分布")

            if stats['type_stats']:
                # 准备图表数据
                types = [item[0] for item in stats['type_stats']]
                counts = [item[1] for item in stats['type_stats']]

                # 创建饼图
                fig = px.pie(
                    values=counts,
                    names=types,
                    title="文件类型分布"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("暂无数据")

        with col2:
            st.markdown("#### 📊 存储空间分布")

            if stats['type_stats']:
                # 准备图表数据
                types = [item[0] for item in stats['type_stats']]
                sizes = [item[2] for item in stats['type_stats']]

                # 创建柱状图
                fig = px.bar(
                    x=types,
                    y=sizes,
                    title="各类型文件存储空间"
                )
                fig.update_layout(
                    xaxis_title="文件类型",
                    yaxis_title="存储空间 (字节)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("暂无数据")

        # 最近上传的文件
        st.markdown("---")
        st.markdown("#### 📅 最近上传的文件")

        if stats['recent_files']:
            recent_df = pd.DataFrame(stats['recent_files'], columns=[
                '文件名', '上传时间', '上传者'
            ])
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("暂无最近上传的文件")

    def render_settings(self):
        """渲染系统设置界面"""
        st.markdown("### ⚙️ 系统设置")

        # 存储设置
        st.markdown("#### 💾 存储设置")

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"**上传目录:** {self.file_manager.upload_dir}")
            st.info(f"**数据库路径:** {self.file_manager.db_path}")

        with col2:
            # 清理操作
            st.markdown("**维护操作:**")

            if st.button("🧹 清理过期文件"):
                self.cleanup_expired_files()

            if st.button("🔄 重建数据库索引"):
                self.rebuild_database_index()

            if st.button("📊 数据库统计"):
                self.show_database_stats()

        # 系统信息
        st.markdown("---")
        st.markdown("#### 🖥️ 系统信息")

        try:
            upload_dir_size = sum(
                f.stat().st_size for f in self.file_manager.upload_dir.rglob('*') if f.is_file()
            )
            db_size = self.file_manager.db_path.stat().st_size

            col1, col2 = st.columns(2)

            with col1:
                st.metric("上传目录大小", self.file_manager.format_file_size(upload_dir_size))
                st.metric("数据库大小", self.file_manager.format_file_size(db_size))

            with col2:
                st.metric("上传目录文件数", len(list(self.file_manager.upload_dir.rglob('*'))))
                st.metric("数据库版本", "SQLite 3")

        except Exception as e:
            st.error(f"获取系统信息失败: {e}")

    def download_file_action(self, file_id: int):
        """下载文件操作"""
        success, message, file_content, filename = self.file_manager.download_file(file_id)

        if success:
            st.download_button(
                label=f"📥 下载 {filename}",
                data=file_content,
                file_name=filename,
                key=f"download_btn_{file_id}"
            )
            st.success(message)
        else:
            st.error(message)

    def edit_file_action(self, file_id: int):
        """编辑文件信息操作"""
        st.markdown(f"#### ✏️ 编辑文件信息 (ID: {file_id})")

        with st.form(f"edit_form_{file_id}"):
            new_description = st.text_area("新描述", key=f"desc_{file_id}")
            new_tags = st.text_input("新标签", key=f"tags_{file_id}")

            if st.form_submit_button("💾 保存更改"):
                success, message = self.file_manager.update_file_info(
                    file_id, new_description, new_tags
                )

                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

    def delete_file_action(self, file_id: int):
        """删除文件操作"""
        if st.button(f"⚠️ 确认删除文件 ID: {file_id}", key=f"confirm_delete_{file_id}"):
            success, message = self.file_manager.delete_file(file_id)

            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    def load_to_data_management(self, file_id: int):
        """加载文件到数据管理模块"""
        success, message, file_content, filename = self.file_manager.download_file(file_id)

        if success:
            try:
                # 根据文件扩展名加载数据
                file_ext = Path(filename).suffix.lower()

                if file_ext == '.csv':
                    df = pd.read_csv(BytesIO(file_content))
                elif file_ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(BytesIO(file_content))
                elif file_ext == '.json':
                    df = pd.read_json(BytesIO(file_content))
                elif file_ext == '.parquet':
                    df = pd.read_parquet(BytesIO(file_content))
                else:
                    st.error(f"不支持的数据文件格式: {file_ext}")
                    return

                # 添加到用户数据
                dataset_name = Path(filename).stem
                if 'user_data' not in st.session_state:
                    st.session_state.user_data = {}

                st.session_state.user_data[dataset_name] = df
                st.success(f"✅ 数据已加载到数据管理模块: {dataset_name}")

            except Exception as e:
                st.error(f"加载数据失败: {e}")
        else:
            st.error(message)

    def cleanup_expired_files(self):
        """清理过期文件"""
        # 这里可以实现清理逻辑
        st.info("🧹 清理功能开发中...")

    def rebuild_database_index(self):
        """重建数据库索引"""
        try:
            conn = sqlite3.connect(self.file_manager.db_path)
            cursor = conn.cursor()

            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON file_records(file_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_upload_time ON file_records(upload_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_type ON file_records(file_type)")

            conn.commit()
            conn.close()

            st.success("✅ 数据库索引重建完成")

        except Exception as e:
            st.error(f"重建索引失败: {e}")

    def show_database_stats(self):
        """显示数据库统计信息"""
        try:
            conn = sqlite3.connect(self.file_manager.db_path)
            cursor = conn.cursor()

            # 获取表信息
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()

            st.markdown("**数据库表信息:**")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cursor.fetchone()[0]
                st.text(f"- {table[0]}: {count} 条记录")

            conn.close()

        except Exception as e:
            st.error(f"获取数据库统计信息失败: {e}")

# 全局实例
file_manager_module = FileManagerModule()
