#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
测试脚本 - 验证数据插入环境是否正常
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_csv_file():
    """测试CSV文件是否存在且格式正确"""
    csv_file = 'data/news_score/news_log.csv'
    print("1. 测试CSV文件...")
    
    if not os.path.exists(csv_file):
        print(f"   ❌ CSV文件不存在: {csv_file}")
        return False
    
    print(f"   ✅ CSV文件存在: {csv_file}")
    
    # 检查文件内容
    with open(csv_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) == 0:
            print("   ❌ CSV文件为空")
            return False
        
        print(f"   ✅ CSV文件包含 {len(lines)} 行数据")
        
        # 检查格式
        first_line = lines[0].strip()
        parts = first_line.split(',')
        if len(parts) != 3:
            print(f"   ❌ CSV格式错误，期望3列，实际{len(parts)}列")
            return False
        
        print("   ✅ CSV格式正确 (user_id,score,content_id)")
        print(f"   示例: {first_line}")
    
    return True

def test_mongodb_connection():
    """测试MongoDB连接"""
    print("\n2. 测试MongoDB连接...")
    
    try:
        from dao.mongo_db import MongoDB
        mongo = MongoDB(db='recommendation')
        
        # 尝试插入一个测试文档
        test_collection = mongo.db_client['test_connection']
        test_doc = {'test': 'connection', 'timestamp': '2024-01-01'}
        result = test_collection.insert_one(test_doc)
        
        # 删除测试文档
        test_collection.delete_one({'_id': result.inserted_id})
        
        print("   ✅ MongoDB连接成功")
        return True
        
    except Exception as e:
        print(f"   ❌ MongoDB连接失败: {e}")
        print("   提示: 请确保MongoDB服务正在运行")
        return False

def test_directory_structure():
    """测试目录结构"""
    print("\n3. 测试目录结构...")
    
    required_dirs = [
        'data/news_score',
        'data/recall_model/CF_model'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ 目录存在: {dir_path}")
        else:
            print(f"   ❌ 目录不存在: {dir_path}")
            all_exist = False
    
    return all_exist

def test_python_dependencies():
    """测试Python依赖"""
    print("\n4. 测试Python依赖...")
    
    required_modules = ['pymongo', 'csv', 'datetime', 'pickle']
    all_available = True
    
    for module_name in required_modules:
        try:
            __import__(module_name)
            print(f"   ✅ 模块可用: {module_name}")
        except ImportError:
            print(f"   ❌ 模块缺失: {module_name}")
            all_available = False
    
    return all_available

def main():
    """主测试函数"""
    print("=== 推荐系统数据插入环境测试 ===\n")
    
    tests = [
        test_csv_file,
        test_mongodb_connection,
        test_directory_structure,
        test_python_dependencies
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{len(tests)} 项测试")
    
    if passed == len(tests):
        print("🎉 所有测试通过！可以安全执行数据插入脚本")
        print("\n推荐执行顺序:")
        print("1. python scripts/insert_likes_data.py  (快速测试)")
        print("2. python scripts/insert_mock_data.py   (完整数据)")
    else:
        print("⚠️  存在问题，请解决后再执行数据插入脚本")
        
        if not test_mongodb_connection():
            print("\n解决方案:")
            print("- 启动MongoDB服务: mongod")
            print("- 检查MongoDB配置")
        
        missing_deps = []
        required_modules = ['pymongo']
        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError:
                missing_deps.append(module_name)
        
        if missing_deps:
            print(f"\n安装缺失依赖: pip install {' '.join(missing_deps)}")

if __name__ == '__main__':
    main() 