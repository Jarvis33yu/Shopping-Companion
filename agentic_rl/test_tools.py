import asyncio
import yaml
from pathlib import Path
from typing import Dict, List, Any
import sys
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agentic_tools import (
    MemSearch,
    MemView,
    MemSummarizeByDate,
    ProductSearch,
    ProductView,
    WebSearch,
    WebVisit
)
from verl.tools.schemas import OpenAIFunctionToolSchema


class ConfigLoader:
    """配置加载器"""

    @staticmethod
    def load_tools_config(yaml_path: str) -> List[Dict[str, Any]]:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('tools', [])

    @staticmethod
    def get_tool_schema(tool_name: str, yaml_path: str) -> Dict[str, Any]:
        tools = ConfigLoader.load_tools_config(yaml_path)
        for tool in tools:
            if tool['class_name'] == tool_name:
                return tool['tool_schema']
        raise ValueError(f"Tool {tool_name} not found")

    @staticmethod
    def get_tool_config(tool_name: str, yaml_path: str) -> Dict[str, Any]:
        tools = ConfigLoader.load_tools_config(yaml_path)
        for tool in tools:
            if tool['class_name'] == tool_name:
                return tool['config']
        raise ValueError(f"Tool {tool_name} not found")


class TestShoppingTools:
    """测试所有购物和检索工具"""

    def setup_method(self):
        self.yaml_path = Path(__file__).parent / "agentic_tools.yaml"
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"agentic_tools.yaml not found")
        print(f"\n✓ 加载配置文件: {self.yaml_path}")
        self.config_loader = ConfigLoader()
        self.conversation_id = "945b2d613b3849140c9176816a9c088b"
        self.question_id = "945b2d613b3849140c9176816a9c088b_1"

    def get_tool_instance(self, tool_class_name: str):
        config = self.config_loader.get_tool_config(tool_class_name, str(self.yaml_path))
        tool_schema_dict = self.config_loader.get_tool_schema(tool_class_name, str(self.yaml_path))
        tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
        
        # 根据工具名称获取对应的工具类
        tool_classes = {
            "agentic_tools.MemSearch": MemSearch,
            "agentic_tools.MemView": MemView,
            "agentic_tools.MemSummarizeByDate": MemSummarizeByDate,
            "agentic_tools.ProductSearch": ProductSearch,
            "agentic_tools.ProductView": ProductView,
            "agentic_tools.WebSearch": WebSearch,
            "agentic_tools.WebVisit": WebVisit,
        }
        
        tool_class = tool_classes.get(tool_class_name)
        if not tool_class:
            raise ValueError(f"Unknown tool class: {tool_class_name}")
        
        return tool_class(config, tool_schema)

    # ==================== 内存搜索工具 ====================
    
    async def test_mem_search_tool(self):
        """测试 MemSearch - 搜索相似记忆"""
        print("\n【测试 1：MemSearch - 搜索相似记忆】")
        tool = self.get_tool_instance("agentic_tools.MemSearch")
        instance_id, _ = await tool.create(
            create_kwargs={
                "conversation_id": self.conversation_id,
                "question_id": self.question_id
                }
        )
        
        parameters = {
            "queries": ["shorts"],
        }
        
        try:
            tool_response, reward, metadata = await tool.execute(instance_id, parameters)
            print(f"✓ MemSearch 执行成功")
            print(f"  返回内容预览: {tool_response.text[:200]}...")
            print(f"当前工具的 reward 是: {reward}")
        except Exception as e:
            print(f"⚠️  执行失败: {str(e)}")
        finally:
            await tool.release(instance_id)

    async def test_mem_view_tool(self):
        """测试 MemView - 查看特定索引的记忆"""
        print("\n【测试 2：MemView - 查看特定索引的记忆】")
        tool = self.get_tool_instance("agentic_tools.MemView")
        instance_id, _ = await tool.create(
            create_kwargs={"conversation_id": self.conversation_id}
        )
        
        parameters = {
            "indices": [4],
        }
        
        try:
            tool_response, reward, metadata = await tool.execute(instance_id, parameters)
            print(f"✓ MemView 执行成功")
            print(f"  返回内容预览: {tool_response.text[:200]}...")
            print(f"当前工具的 reward 是: {reward}")
        except Exception as e:
            print(f"⚠️  执行失败: {str(e)}")
        finally:
            await tool.release(instance_id)

    async def test_mem_summarize_by_date_tool(self):
        """测试 MemSummarizeByDate - 按日期范围汇总记忆"""
        print("\n【测试 3：MemSummarizeByDate - 按日期范围汇总记忆】")
        tool = self.get_tool_instance("agentic_tools.MemSummarizeByDate")
        instance_id, _ = await tool.create(
            create_kwargs={"conversation_id": self.conversation_id}
        )
        
        parameters = {
            "start_date": "2025-05-01",
            "offset": 30,  # 30天
            "goal": "iphone",
        }
        
        try:
            tool_response, reward, metadata = await tool.execute(instance_id, parameters)
            print(f"✓ MemSummarizeByDate 执行成功")
            print(f"  返回内容预览: {tool_response.text[:200]}...")
            print(f"当前工具的 reward 是: {reward}")
        except Exception as e:
            print(f"⚠️  执行失败: {str(e)}")
        finally:
            await tool.release(instance_id)

    # ==================== 产品工具 ====================

    async def test_product_search_tool(self):
        """测试 ProductSearch - 搜索产品"""
        print("\n【测试 4：ProductSearch - 搜索产品】")
        tool = self.get_tool_instance("agentic_tools.ProductSearch")
        instance_id, _ = await tool.create()
        
        parameters = {
            "query": "laptop sink stopper onion bundle",
            "shop_id": None,  # 可选
            "price": "",  # 可选，格式: "min-max"
        }
        
        try:
            tool_response, reward, metadata = await tool.execute(instance_id, parameters)
            print(f"✓ ProductSearch 执行成功")
            print(f"  返回内容预览: {tool_response.text[:200]}...")
            print(f"当前工具的 reward 是: {reward}")
        except Exception as e:
            print(f"⚠️  执行失败: {str(e)}")
        finally:
            await tool.release(instance_id)

    async def test_product_view_tool(self):
        """测试 ProductView - 查看产品详情"""
        print("\n【测试 5：ProductView - 查看产品详情】")
        tool = self.get_tool_instance("agentic_tools.ProductView")
        instance_id, _ = await tool.create()
        
        parameters = {
            "product_ids": ["5250901107", "5250923607", "5251048584"],  # 产品ID列表
        }
        
        try:
            tool_response, reward, metadata = await tool.execute(instance_id, parameters)
            print(f"✓ ProductView 执行成功")
            print(f"  返回内容预览: {tool_response.text[:200]}...")
            print(f"当前工具的 reward 是: {reward}")
        except Exception as e:
            print(f"⚠️  执行失败: {str(e)}")
        finally:
            await tool.release(instance_id)

    # ==================== 网页工具 ====================

    async def test_web_search_tool(self):
        """测试 WebSearch - 网页搜索"""
        print("\n【测试 6：WebSearch - 网页搜索】")
        tool = self.get_tool_instance("agentic_tools.WebSearch")
        instance_id, _ = await tool.create()
        
        parameters = {
            "queries": ["best laptops 2024", "laptop reviews"],
        }
        
        try:
            tool_response, reward, metadata = await tool.execute(instance_id, parameters)
            print(f"✓ WebSearch 执行成功")
            print(f"  返回内容预览: {tool_response.text[:200]}...")
        except Exception as e:
            print(f"⚠️  执行失败: {str(e)}")
        finally:
            await tool.release(instance_id)

    async def test_web_visit_tool(self):
        """测试 WebVisit - 访问网页并总结"""
        print("\n【测试 7：WebVisit - 访问网页并总结】")
        tool = self.get_tool_instance("agentic_tools.WebVisit")
        instance_id, _ = await tool.create()
        
        parameters = {
            "urls": [
                "https://www.baidu.com/",
                "https://www.runoob.com/python/python-tutorial.html",
            ],
            "goal": "Search for Python tutorials",
        }
        
        try:
            tool_response, reward, metadata = await tool.execute(instance_id, parameters)
            print(f"✓ WebVisit 执行成功")
            print(f"  返回内容预览: {tool_response.text[:]}...")
        except Exception as e:
            print(f"⚠️  执行失败: {str(e)}")
        finally:
            await tool.release(instance_id)
    
    async def test_mem_search_concurrent(self, task_id: int, delay: float = 0):
        """单个并发任务 - MemSearch"""
        if delay > 0:
            await asyncio.sleep(delay)
        
        tool = self.get_tool_instance("agentic_tools.MemSearch")
        instance_id, _ = await tool.create(
            create_kwargs={"conversation_id": self.conversation_id}
        )
        
        parameters = {
            "queries": [f"query_{task_id}_1", f"query_{task_id}_2", f"query_{task_id}_3"],
        }
        
        start_time = time.time()
        try:
            tool_response, reward, metadata = await tool.execute(instance_id, parameters)
            elapsed_time = time.time() - start_time
            print(f"  ✓ 任务 {task_id} 完成 (耗时: {elapsed_time:.2f}s)")
            return {
                "task_id": task_id,
                "status": "success",
                "elapsed_time": elapsed_time,
                "response_length": len(tool_response.text)
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"  ✗ 任务 {task_id} 失败 (耗时: {elapsed_time:.2f}s): {str(e)}")
            return {
                "task_id": task_id,
                "status": "failed",
                "elapsed_time": elapsed_time,
                "error": str(e)
            }
        finally:
            await tool.release(instance_id)

    async def test_mem_search_parallel_tasks(self, num_tasks: int = 5):
        """测试 MemSearch 多并发"""
        print(f"\n【测试 MemSearch 并发 - {num_tasks} 个并发任务】")
        
        start_time = time.time()
        
        # 方法1: 同时启动所有任务
        tasks = [
            self.test_mem_search_concurrent(task_id=i)
            for i in range(num_tasks)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # 统计结果
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        failed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "failed")
        
        print(f"\n  📊 并发结果统计:")
        print(f"    - 总任务数: {num_tasks}")
        print(f"    - 成功: {successful}")
        print(f"    - 失败: {failed}")
        print(f"    - 总耗时: {total_time:.2f}s")
        
        if successful > 0:
            avg_time = sum(r['elapsed_time'] for r in results if isinstance(r, dict) and r.get("status") == "success") / successful
            print(f"    - 平均耗时: {avg_time:.2f}s")
        
        return results

    async def test_mem_search_sequential_vs_concurrent(self, num_tasks: int = 5):
        """对比顺序执行 vs 并发执行"""
        print(f"\n【对比测试：顺序 vs 并发 ({num_tasks} 个任务)】")
        
        # 顺序执行
        print("\n  📌 顺序执行:")
        start_time = time.time()
        for i in range(num_tasks):
            await self.test_mem_search_concurrent(task_id=i)
        sequential_time = time.time() - start_time
        print(f"  总耗时: {sequential_time:.2f}s")
        
        # 并发执行
        print("\n  📌 并发执行:")
        start_time = time.time()
        tasks = [
            self.test_mem_search_concurrent(task_id=i)
            for i in range(num_tasks)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        concurrent_time = time.time() - start_time
        print(f"  总耗时: {concurrent_time:.2f}s")
        
        # 性能对比
        print(f"\n  ⚡ 性能提升: {sequential_time / concurrent_time:.2f}x (顺序时间/并发时间)")

    async def test_mem_search_stress_test(self, num_tasks: int = 20):
        """压力测试 - 大量并发请求"""
        print(f"\n【压力测试：{num_tasks} 个并发任务】")
        
        start_time = time.time()
        tasks = [
            self.test_mem_search_concurrent(task_id=i)
            for i in range(num_tasks)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # 统计结果
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        failed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "failed")
        
        print(f"\n  📊 压力测试结果:")
        print(f"    - 总任务数: {num_tasks}")
        print(f"    - 成功: {successful}")
        print(f"    - 失败: {failed}")
        print(f"    - 成功率: {successful / num_tasks * 100:.1f}%")
        print(f"    - 总耗时: {total_time:.2f}s")
        print(f"    - 吞吐量: {num_tasks / total_time:.2f} 任务/秒")
        
        return results

    # ==================== 错误测试 ====================

    async def test_error_handling(self):
        """测试错误处理"""
        print("\n【测试 8：错误处理 - 缺少必要参数】")
        
        # 测试 MemSearch 缺少 queries
        tool = self.get_tool_instance("agentic_tools.MemSearch")
        instance_id, _ = await tool.create(
            create_kwargs={"conversation_id": self.conversation_id}
        )
        
        parameters = {
            # 缺少 "queries" 参数
        }
        
        try:
            tool_response, reward, metadata = await tool.execute(instance_id, parameters)
            if "error" in tool_response.text:
                print(f"✓ 正确捕获错误: {tool_response.text}")
            else:
                print(f"⚠️  未能正确处理缺失参数")
        except Exception as e:
            print(f"✓ 捕获异常: {str(e)}")
        finally:
            await tool.release(instance_id)

    async def test_product_view_concurrent(self, task_id: int, delay: float = 0):
        """单个并发任务 - ProductView"""
        if delay > 0:
            await asyncio.sleep(delay)
        
        tool = self.get_tool_instance("agentic_tools.ProductView")
        instance_id, _ = await tool.create()
        
        # 不同任务使用不同的产品ID组合
        product_id_sets = [
            ["5250901107", "5250923607", "5251048584"],
            ["5250901107", "5250923607"],
            ["5251048584"],
            ["5250901107"],
            ["5250923607"],
            ["5250901107", "5250923607", "5251048584", "5250901108"],
            ["5250923607", "5251048584"],
            ["5250901107", "5251048584"],
            ["5250901108"],
            ["5250901109"],
        ]
        
        product_ids = product_id_sets[task_id % len(product_id_sets)]
        
        parameters = {
            "product_ids": product_ids,
        }
        
        start_time = time.time()
        try:
            tool_response, reward, metadata = await tool.execute(instance_id, parameters)
            elapsed_time = time.time() - start_time
            response_length = len(tool_response.text) if tool_response.text else 0
            
            # 检查是否有错误
            is_error = "error" in tool_response.text.lower()
            status = "failed" if is_error else "success"
            
            print(f"  ✓ 任务 {task_id} 完成 (产品ID: {product_ids}, 耗时: {elapsed_time:.2f}s, 响应长度: {response_length})")
            
            return {
                "task_id": task_id,
                "product_ids": product_ids,
                "status": status,
                "elapsed_time": elapsed_time,
                "response_length": response_length,
                "is_error": is_error
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"  ✗ 任务 {task_id} 失败 (产品ID: {product_ids}, 耗时: {elapsed_time:.2f}s): {str(e)}")
            return {
                "task_id": task_id,
                "product_ids": product_ids,
                "status": "failed",
                "elapsed_time": elapsed_time,
                "response_length": 0,
                "error": str(e),
                "is_error": True
            }
        finally:
            await tool.release(instance_id)

    async def test_product_view_parallel_tasks(self, num_tasks: int = 5):
        """测试 ProductView 多并发"""
        print(f"\n【测试 ProductView 并发 - {num_tasks} 个并发任务】")
        
        start_time = time.time()
        
        tasks = [
            self.test_product_view_concurrent(task_id=i)
            for i in range(num_tasks)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # 统计结果
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        failed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "failed")
        
        print(f"\n  📊 并发结果统计:")
        print(f"    - 总任务数: {num_tasks}")
        print(f"    - 成功: {successful}")
        print(f"    - 失败: {failed}")
        print(f"    - 成功率: {successful / num_tasks * 100:.1f}%")
        print(f"    - 总耗时: {total_time:.2f}s")
        
        if successful > 0:
            avg_time = sum(r['elapsed_time'] for r in results if isinstance(r, dict) and r.get("status") == "success") / successful
            print(f"    - 平均耗时: {avg_time:.2f}s")
            print(f"    - 吞吐量: {successful / total_time:.2f} 任务/秒")
            
            # 响应大小统计
            response_sizes = [r['response_length'] for r in results if isinstance(r, dict) and r.get("status") == "success"]
            if response_sizes:
                print(f"    - 平均响应大小: {sum(response_sizes) / len(response_sizes):.0f} 字节")
        
        return results

    async def test_product_view_sequential_vs_concurrent(self, num_tasks: int = 5):
        """对比 ProductView 顺序执行 vs 并发执行"""
        print(f"\n【对比测试：ProductView 顺序 vs 并发 ({num_tasks} 个任务)】")
        
        # 顺序执行
        print("\n  📌 顺序执行:")
        start_time = time.time()
        for i in range(num_tasks):
            await self.test_product_view_concurrent(task_id=i)
        sequential_time = time.time() - start_time
        print(f"  总耗时: {sequential_time:.2f}s")
        
        # 并发执行
        print("\n  📌 并发执行:")
        start_time = time.time()
        tasks = [
            self.test_product_view_concurrent(task_id=i)
            for i in range(num_tasks)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        concurrent_time = time.time() - start_time
        print(f"  总耗时: {concurrent_time:.2f}s")
        
        # 性能对比
        if concurrent_time > 0:
            speedup = sequential_time / concurrent_time
            print(f"\n  ⚡ 性能提升: {speedup:.2f}x (顺序时间/并发时间)")

    async def test_product_view_stress_test(self, num_tasks: int = 20):
        """压力测试 ProductView - 大量并发请求"""
        print(f"\n【ProductView 压力测试：{num_tasks} 个并发任务】")
        
        start_time = time.time()
        tasks = [
            self.test_product_view_concurrent(task_id=i)
            for i in range(num_tasks)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # 统计结果
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        failed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "failed")
        
        print(f"\n  📊 压力测试结果:")
        print(f"    - 总任务数: {num_tasks}")
        print(f"    - 成功: {successful}")
        print(f"    - 失败: {failed}")
        print(f"    - 成功率: {successful / num_tasks * 100:.1f}%")
        print(f"    - 总耗时: {total_time:.2f}s")
        print(f"    - 吞吐量: {num_tasks / total_time:.2f} 任务/秒")
        
        # 详细的耗时统计
        if successful > 0:
            times = [r['elapsed_time'] for r in results if isinstance(r, dict) and r.get("status") == "success"]
            print(f"    - 最快: {min(times):.2f}s")
            print(f"    - 最慢: {max(times):.2f}s")
            print(f"    - 平均: {sum(times) / len(times):.2f}s")
            
            # 响应大小统计
            response_sizes = [r['response_length'] for r in results if isinstance(r, dict) and r.get("status") == "success"]
            if response_sizes:
                print(f"    - 总响应大小: {sum(response_sizes) / 1024:.2f} KB")
                print(f"    - 平均响应大小: {sum(response_sizes) / len(response_sizes):.0f} 字节")
        
        return results

    async def test_product_search_concurrent(self, task_id: int, delay: float = 0):
        """单个并发任务 - ProductSearch"""
        if delay > 0:
            await asyncio.sleep(delay)
        
        tool = self.get_tool_instance("agentic_tools.ProductSearch")
        instance_id, _ = await tool.create()
        
        # 不同任务使用不同的查询词
        queries = [
            "laptop",
            "phone",
            "tablet",
            "headphones",
            "smartwatch",
            "camera",
            "speaker",
            "monitor",
            "keyboard",
            "mouse"
        ]
        
        query = queries[task_id % len(queries)]
        
        parameters = {
            "query": query,
            "shop_id": None,
            "price": "",
        }
        
        start_time = time.time()
        try:
            tool_response, reward, metadata = await tool.execute(instance_id, parameters)
            elapsed_time = time.time() - start_time
            print(f"  ✓ 任务 {task_id} 完成 (查询: '{query}', 耗时: {elapsed_time:.2f}s)")
            return {
                "task_id": task_id,
                "query": query,
                "status": "success",
                "elapsed_time": elapsed_time,
                "response_length": len(tool_response.text)
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"  ✗ 任务 {task_id} 失败 (查询: '{query}', 耗时: {elapsed_time:.2f}s): {str(e)}")
            return {
                "task_id": task_id,
                "query": query,
                "status": "failed",
                "elapsed_time": elapsed_time,
                "error": str(e)
            }
        finally:
            await tool.release(instance_id)

    async def test_product_search_parallel_tasks(self, num_tasks: int = 5):
        """测试 ProductSearch 多并发"""
        print(f"\n【测试 ProductSearch 并发 - {num_tasks} 个并发任务】")
        
        start_time = time.time()
        
        tasks = [
            self.test_product_search_concurrent(task_id=i)
            for i in range(num_tasks)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # 统计结果
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        failed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "failed")
        
        print(f"\n  📊 并发结果统计:")
        print(f"    - 总任务数: {num_tasks}")
        print(f"    - 成功: {successful}")
        print(f"    - 失败: {failed}")
        print(f"    - 成功率: {successful / num_tasks * 100:.1f}%")
        print(f"    - 总耗时: {total_time:.2f}s")
        
        if successful > 0:
            avg_time = sum(r['elapsed_time'] for r in results if isinstance(r, dict) and r.get("status") == "success") / successful
            print(f"    - 平均耗时: {avg_time:.2f}s")
            print(f"    - 吞吐量: {successful / total_time:.2f} 任务/秒")
        
        return results

    async def test_product_search_sequential_vs_concurrent(self, num_tasks: int = 5):
        """对比 ProductSearch 顺序执行 vs 并发执行"""
        print(f"\n【对比测试：ProductSearch 顺序 vs 并发 ({num_tasks} 个任务)】")
        
        # 顺序执行
        print("\n  📌 顺序执行:")
        start_time = time.time()
        for i in range(num_tasks):
            await self.test_product_search_concurrent(task_id=i)
        sequential_time = time.time() - start_time
        print(f"  总耗时: {sequential_time:.2f}s")
        
        # 并发执行
        print("\n  📌 并发执行:")
        start_time = time.time()
        tasks = [
            self.test_product_search_concurrent(task_id=i)
            for i in range(num_tasks)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        concurrent_time = time.time() - start_time
        print(f"  总耗时: {concurrent_time:.2f}s")
        
        # 性能对比
        if concurrent_time > 0:
            speedup = sequential_time / concurrent_time
            print(f"\n  ⚡ 性能提升: {speedup:.2f}x (顺序时间/并发时间)")

    async def test_product_search_stress_test(self, num_tasks: int = 20):
        """压力测试 ProductSearch - 大量并发请求"""
        print(f"\n【ProductSearch 压力测试：{num_tasks} 个并发任务】")
        
        start_time = time.time()
        tasks = [
            self.test_product_search_concurrent(task_id=i)
            for i in range(num_tasks)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # 统计结果
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        failed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "failed")
        
        print(f"\n  📊 压力测试结果:")
        print(f"    - 总任务数: {num_tasks}")
        print(f"    - 成功: {successful}")
        print(f"    - 失败: {failed}")
        print(f"    - 成功率: {successful / num_tasks * 100:.1f}%")
        print(f"    - 总耗时: {total_time:.2f}s")
        print(f"    - 吞吐量: {num_tasks / total_time:.2f} 任务/秒")
        
        # 详细的耗时统计
        if successful > 0:
            times = [r['elapsed_time'] for r in results if isinstance(r, dict) and r.get("status") == "success"]
            print(f"    - 最快: {min(times):.2f}s")
            print(f"    - 最慢: {max(times):.2f}s")
            print(f"    - 平均: {sum(times) / len(times):.2f}s")
        
        return results




    async def run_all_tests(self):
        """运行所有测试"""
        try:
            # # 并发测试
            # await self.test_mem_search_parallel_tasks(num_tasks=5)
            # await self.test_mem_search_sequential_vs_concurrent(num_tasks=5)
            # # await self.test_mem_search_stress_test(num_tasks=1000)
            # await self.test_product_search_parallel_tasks(num_tasks=5)
            # await self.test_product_search_sequential_vs_concurrent(num_tasks=5)
            # await self.test_product_search_stress_test(num_tasks=1000)
            # await self.test_product_view_parallel_tasks(num_tasks=5)
            # await self.test_product_view_sequential_vs_concurrent(num_tasks=5)
            # await self.test_product_view_stress_test(num_tasks=1000)
            # # 内存工具测试
            await self.test_mem_search_tool()
            await self.test_mem_view_tool()
            # await self.test_mem_summarize_by_date_tool()
            
            # # 产品工具测试
            # await self.test_product_search_tool()
            # await self.test_product_view_tool()
            
            # # # 网页工具测试
            # await self.test_web_search_tool()
            # await self.test_web_visit_tool()
            
            # # 错误处理测试
            # await self.test_error_handling()
            
            # print("\n" + "="*60)
            # print("✓ 所有工具测试完成！")
            # print("="*60 + "\n")

        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            raise


async def main():
    tester = TestShoppingTools()
    tester.setup_method()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
