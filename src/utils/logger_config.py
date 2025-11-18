# utils/logger_config.py
import os
import sys
import time
from functools import wraps
from pathlib import Path

from loguru import logger


class LoggerConfig:
    """
    日志配置类 - 支持多种开关方式
    """

    # ============ 方式1: 类属性（最简单，性能最好） ============
    ENABLE_LOGGING = True  # 主开关
    ENABLE_FILE_LOG = True  # 文件日志开关
    ENABLE_CONSOLE_LOG = True  # 控制台日志开关

    # ============ 日志配置 ============
    LOG_LEVEL = "DEBUG"  # TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
    LOG_DIR = Path("logs")
    LOG_ROTATION = "100 MB"  # 文件大小轮转
    LOG_RETENTION = "30 days"  # 保留时间
    LOG_COMPRESSION = "zip"  # 压缩格式

    # ============ 日志格式 ============
    @staticmethod
    def format_extra(record):
        """格式化 extra 字段，只在有内容时显示"""
        extra = record["extra"]
        if extra:
            return " | " + " | ".join(f"{k}={v}" for k, v in extra.items())
        return ""

    CONSOLE_FORMAT = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    FILE_FORMAT = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    _initialized = False
    _handlers: list = []  # 保存handler ID，用于后续移除

    @classmethod
    def init_logger(cls):
        """
        初始化logger（只执行一次）
        """
        if cls._initialized:
            return logger

        # 移除默认handler
        logger.remove()

        if not cls.ENABLE_LOGGING:
            # 如果日志完全关闭，添加一个null handler
            logger.add(
                lambda _: None, level="CRITICAL", format="{message}"  # 不做任何事
            )
            cls._initialized = True
            return logger

        # 创建日志目录
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)

        # ============ 控制台日志 ============
        if cls.ENABLE_CONSOLE_LOG:

            def console_formatter(record):
                """控制台日志格式化器"""
                extra_str = cls.format_extra(record)
                record["extra_formatted"] = extra_str
                return cls.CONSOLE_FORMAT + "<yellow>{extra_formatted}</yellow>\n"

            handler_id = logger.add(
                sys.stdout,
                format=console_formatter,
                level=cls.LOG_LEVEL,
                colorize=True,
                enqueue=False,  # 异步写入,提升性能
                backtrace=True,
                diagnose=True,
            )
            cls._handlers.append(handler_id)

        # ============ 文件日志 ============
        if cls.ENABLE_FILE_LOG:

            def file_formatter(record):
                """文件日志格式化器"""
                extra_str = cls.format_extra(record)
                record["extra_formatted"] = extra_str
                return cls.FILE_FORMAT + "{extra_formatted}\n"

            # 所有日志
            handler_id = logger.add(
                cls.LOG_DIR / "app_{time:YYYY-MM-DD}.log",
                format=file_formatter,
                level=cls.LOG_LEVEL,
                rotation=cls.LOG_ROTATION,
                retention=cls.LOG_RETENTION,
                compression=cls.LOG_COMPRESSION,
                enqueue=True,  # 异步写入
                encoding="utf-8",
            )
            cls._handlers.append(handler_id)

            # 错误日志单独记录
            handler_id = logger.add(
                cls.LOG_DIR / "error_{time:YYYY-MM-DD}.log",
                format=file_formatter,
                level="ERROR",
                rotation=cls.LOG_ROTATION,
                retention=cls.LOG_RETENTION,
                compression=cls.LOG_COMPRESSION,
                enqueue=True,
                encoding="utf-8",
            )
            cls._handlers.append(handler_id)

        cls._initialized = True
        logger.success("Logger 初始化成功！")
        return logger

    @classmethod
    def enable(cls):
        """启用日志"""
        cls.ENABLE_LOGGING = True
        cls.reload()

    @classmethod
    def disable(cls):
        """禁用日志（性能最优）"""
        cls.ENABLE_LOGGING = False
        cls.reload()

    @classmethod
    def reload(cls):
        """重新加载日志配置"""
        cls._initialized = False
        cls._handlers.clear()
        logger.remove()  # 移除所有handlers
        cls.init_logger()

    @classmethod
    def load_from_env(cls):
        """
        从环境变量加载配置
        """
        cls.ENABLE_LOGGING = os.getenv("LOG_ENABLE", "true").lower() == "true"
        cls.ENABLE_FILE_LOG = os.getenv("LOG_FILE_ENABLE", "true").lower() == "true"
        cls.ENABLE_CONSOLE_LOG = (
            os.getenv("LOG_CONSOLE_ENABLE", "true").lower() == "true"
        )
        cls.LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

        log_dir = os.getenv("LOG_DIR", "logs")
        cls.LOG_DIR = Path(log_dir)

    @classmethod
    def load_from_config(cls, config_dict):
        """
        从配置字典加载
        """
        cls.ENABLE_LOGGING = config_dict.get("enable_logging", True)
        cls.ENABLE_FILE_LOG = config_dict.get("enable_file_log", True)
        cls.ENABLE_CONSOLE_LOG = config_dict.get("enable_console_log", True)
        cls.LOG_LEVEL = config_dict.get("log_level", "DEBUG")
        cls.LOG_DIR = Path(config_dict.get("log_dir", "logs"))


# ============ 性能优化装饰器 ============
def log_execution_time(func):
    """
    记录函数执行时间（只在日志开启时生效）
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not LoggerConfig.ENABLE_LOGGING:
            return func(*args, **kwargs)

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        logger.debug(
            f"Function '{func.__name__}' 执行时间为 "
            f"{(end_time - start_time) * 1000:.2f}ms"
        )
        return result

    return wrapper


def catch_exceptions(func):
    """
    捕获并记录异常
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if LoggerConfig.ENABLE_LOGGING:
                logger.exception(f"Exception in {func.__name__}: {e}")
            raise

    return wrapper


# ============ 便捷函数 ============
def get_logger():
    """获取配置好的logger实例"""
    if not LoggerConfig._initialized:
        LoggerConfig.init_logger()
    return logger


# 自动初始化
LoggerConfig.init_logger()
