import re
import os

# --- 設定 ---
# 入力ファイル名（VizieRからダウンロードしたファイル）
input_filename = 'vizier_votable.tsv'
# 出力するCSVファイル名
output_filename = 'yield_data.csv'
# ----------------

def extract_and_convert_csv():
    """
    VizieRのVOTableファイルからCSVデータを抽出し、
    カンマ区切りのCSVファイルとして保存します。
    """
    try:
        # 入力ファイルを読み込む
        with open(input_filename, 'r', encoding='utf-8') as f:
            content = f.read()

        # 正規表現を使って <![CDATA[...]]> の中身を検索して抽出する
        # re.DOTALL は改行文字もマッチさせるためのオプション
        match = re.search(r'<!\[CDATA\[(.*?)\]\]>', content, re.DOTALL)

        if match:
            # 抽出したCSVデータ部分を取得
            csv_data = match.group(1).strip() # 前後の不要な空白や改行を削除

            # データを1行ずつに分割し、各行を処理
            processed_lines = []
            for line in csv_data.split('\n'):
                # 行頭の空白を削除し、区切り文字のセミコロン(;)をカンマ(,)に置換
                cleaned_line = line.lstrip().replace(';', ',')
                processed_lines.append(cleaned_line)

            # 処理した行を改行で連結して最終的なCSVコンテンツを作成
            final_csv_content = '\n'.join(processed_lines)

            # 結果を新しいCSVファイルに書き込む
            with open(output_filename, 'w', encoding='utf-8') as f_out:
                f_out.write(final_csv_content)

            print(f"✅ 成功: CSVデータを抽出し、'{output_filename}' に保存しました。")
            print(f"   場所: {os.path.abspath(output_filename)}")

        else:
            print("❌ エラー: ファイル内にCSVデータブロック（<![CDATA[...]]>）が見つかりませんでした。")

    except FileNotFoundError:
        print(f"❌ エラー: ファイル '{input_filename}' が見つかりません。")
        print("   スクリプトと同じディレクトリにファイルを置いてください。")
    except Exception as e:
        print(f"❌ 予期せぬエラーが発生しました: {e}")

# スクリプトを実行
if __name__ == "__main__":
    extract_and_convert_csv()