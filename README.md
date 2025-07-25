# AI-System-Course

「幸福スコアの予測モデル」というテーマで機械学習分析を行いました。使用したデータは「World Happiness Report」で、GDPや健康、自由度など7つの要因を特徴量として、幸福スコアを回帰モデルで予測しています。データ前処理では、外れ値をIQR法で除去し、線形回帰モデルには標準化を施しました。線形回帰、ランダムフォレスト、XGBoostの3種類のモデルを比較し、交差検証も行った結果、ランダムフォレストが最も安定して良い精度を出しました。また、ランダムフォレストについては木の本数をハイパーパラメータとして調整し、300本が最適であることが分かりました。今回の分析では、各国の幸福に影響を与える特徴をモデルを通じて定量的に把握でき、予測精度も非常に高い結果となりました。

「睡眠と健康の関係分析」というテーマで、睡眠時間や生活習慣と睡眠障害との関係を探索的に分析しました。使用データは「Sleep health and lifestyle dataset」で、年齢、BMI、運動習慣、職業などの特徴量と睡眠障害の有無との関係を、棒グラフや箱ひげ図を用いて可視化しました。特に、運動習慣や職業、年齢層によって睡眠障害の発生傾向に違いが見られ、健康と生活習慣の関連性を視覚的に明らかにすることができました。

ペンギンの種別（二値分類）を予測する機械学習モデルを構築しました。特徴量には体長・体重・くちばしの長さなどを使用し、XGBoostを用いてバイナリ分類モデルを作成しました。前処理ではカテゴリ変数のエンコーディングを行い、データを学習・検証・テストに分割。モデルはAWS SageMaker上で学習させ、推論用エンドポイントをデプロイ。AUCや混同行列で評価を行い、高い分類精度を確認しました。クラウドを用いた一連の機械学習プロセスを実践的に学ぶことができました。

「顔画像の一致確認」というテーマで、AWSのRekognitionサービスを利用した画像認識プログラムを作成しました。2枚の人物画像を比較し、一致する顔を検出する処理を実装しました。boto3を用いてクラウドAPIと連携し、一致度（類似度スコア）や信頼度、バウンディングボックス情報を取得し、検出結果を赤枠で可視化しました。クラウドAIサービスを活用した画像処理の流れを実践的に学ぶことができました。

「歯科予約管理」というテーマで、Amazon LexとLambdaを活用した会話型チャットボットを実装しました。ユーザーの入力に応じて予約内容（種類・日付・時間）を順次確認し、条件に合った予約枠を提示・確定するロジックを構築。スロットのバリデーションや、日付・時間の自動候補生成、可用性管理なども実装し、自然な対話フローで予約処理を実現しました。クラウド上でのAI対話処理と業務ロジックの連携を実践できたプロジェクトです。

「表情認識システム」というテーマで、画像処理とディープラーニングを組み合わせた感情分類モデルを構築しました。使用したデータは顔表情に特化した「FER-2013」データセットで、怒り・嫌悪・恐れ・幸福・悲しみ・驚き・無表情の7種類の感情をCNNモデルにより分類しています。モデル構築には TensorFlow/Keras を用い、学習済みモデルを .keras 形式で保存。推論では、OpenCV を利用してHaar Cascadeによる顔検出を行い、検出された顔部分をグレースケール化・リサイズ・正規化したうえでモデルに入力します。システムはColab上でもローカルでも動作するように整備されており、アップロードされた画像から感情を判定し、その結果を画像上に表示するとともに、CSVファイルに保存可能です。また、顔検出や前処理などは .py モジュールに分離し、機能を関数化・再利用可能な構造としました。今回の開発を通じて、画像分類タスクにおける前処理・モデル推論・結果の可視化や保存処理を一連のワークフローとして構築する経験が得られました。特に、複数画像・複数顔の処理、精度の高い推論処理、およびユーザー向け出力形式の整備に注力しました。


