#!/usr/bin/env python3
"""
Colab環境用セットアップスクリプト
Fast-dLLMとLongLLaDAリポジトリを自動的にクローンします
"""

import os
import subprocess
import sys


def run_command(cmd, description=""):
    """コマンドを実行し、結果を表示"""
    print(f"🔄 {description}")
    print(f"💻 実行: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, check=True,
                                capture_output=True, text=True)
        if result.stdout:
            print(f"✅ 成功: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ エラー: {e.stderr.strip()}")
        return False


def setup_repositories():
    """Fast-dLLMとLongLLaDAリポジトリをセットアップ"""
    current_dir = os.getcwd()
    print(f"📍 現在のディレクトリ: {current_dir}")

    repositories = [
        {
            "name": "Fast-dLLM",
            "url": "https://github.com/NVlabs/Fast-dLLM.git",
            "dir": "Fast-dLLM"
        },
        {
            "name": "LongLLaDA",
            "url": "https://github.com/OpenMOSS/LongLLaDA.git",
            "dir": "LongLLaDA"
        }
    ]

    for repo in repositories:
        print(f"\n🔍 {repo['name']} のセットアップ中...")

        if os.path.exists(repo['dir']):
            print(f"✅ {repo['dir']} は既に存在します")
            # 既存の場合はプル
            success = run_command(
                f"cd {repo['dir']} && git pull",
                f"{repo['name']} を更新中"
            )
        else:
            # 新規クローン
            success = run_command(
                f"git clone {repo['url']} {repo['dir']}",
                f"{repo['name']} をクローン中"
            )

        if success:
            print(f"✅ {repo['name']} のセットアップ完了")
        else:
            print(f"❌ {repo['name']} のセットアップ失敗")
            return False

    return True


def verify_setup():
    """セットアップが正しく完了したか確認"""
    print(f"\n🔍 セットアップ確認中...")

    required_files = [
        "Fast-dLLM/llada/model/modeling_llada.py",
        "Fast-dLLM/llada/generate.py",
        "LongLLaDA/llada/llada_generate.py"
    ]

    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} が見つかりません")
            all_good = False

    return all_good


def main():
    """メイン実行関数"""
    print("🚀 Fast-dLLM × LongLLaDA Colab セットアップ")
    print("=" * 50)

    # 必要なパッケージのインストール
    print("📦 必要なパッケージをインストール中...")
    packages = ["transformers", "torch", "accelerate", "einops"]
    for package in packages:
        run_command(f"pip install {package}", f"{package} をインストール中")

    # リポジトリのセットアップ
    if setup_repositories():
        print("\n🎉 リポジトリのセットアップ完了！")
    else:
        print("\n❌ リポジトリのセットアップに失敗しました")
        sys.exit(1)

    # セットアップの確認
    if verify_setup():
        print("\n✅ 全てのセットアップが完了しました！")
        print("📝 これで integrated_generation.py が正しく動作するはずです。")
    else:
        print("\n⚠️  一部のファイルが見つかりません。手動で確認してください。")


if __name__ == "__main__":
    main()
