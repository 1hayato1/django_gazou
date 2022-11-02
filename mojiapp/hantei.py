

#### １．各種インポート #### 

import os
import cv2
import numpy as np
from imutils import contours
import matplotlib.pyplot as plt
import japanize_matplotlib
import glob
from natsort import natsorted
from keras.models import load_model


#### ２．OCRする画像 #### 

input_file = 'H5_tegaki.png' # ここを変更
                             # OCRしたい画像のファイル名を入力します


#### ３．各種設定 ####

# 横書き・縦書きの設定
# 横書きの文字領域検出・縦書きの文字領域検出の選択
horizontal = 0
vertical = 1
OCR_mode = 0 # ここを変更。
             # 横書きは「horizontal」または「0」・縦書きは「vertical」または「1」を入力


# 画像判定用の設定
# 学習済みモデルの読み込み
model = load_model('ETL8-model.h5') # ここを変更
                               # 学習済みモデルの画像のカラー設定には「モノクロ・グレースケール」「カラー」があります

# 画像のカラー設定
# 学習済みモデルと同じ画像のカラー設定
color_setting = 3 # ここを変更
                  # モノクロ・グレースケールの場合は「1」。カラーの場合は「3」 

# 画像のサイズ設定
# 学習済みモデルと同じサイズを指定
image_width = 32  # ここを変更。使用する学習済みモデルと同じwidth（横幅）を指定
image_height = 32 # ここを変更。使用する学習済みモデルと同じheight（縦の高さ）を指定


# 膨張処理の設定
# OCRしたい画像に合わせて微調整が必要（文字の太さ・文字の線の間隔・文字の間隔などが影響します）
#【横書き】大まかな文字領域の検出（ブロック検出）のための膨張処理（カーネルサイズ・膨張処理回数）の設定
block_horizontal_kernel_hight = 5  # カーネルの縦の高さ
block_horizontal_kernel_width = 5  # カーネルの横の幅
block_horizontal_iterations = 5    # 膨張処理回数

#【縦書き】大まかな文字領域の検出（ブロック検出）のための膨張処理（カーネルサイズ・膨張処理回数）の設定
block_vertical_kernel_hight = 5  # カーネルの縦の高さ
block_vertical_kernel_width = 5  # カーネルの横の幅
block_vertical_iterations = 9    # 膨張処理回数


#【横書き】行領域の検出（行検出）のための膨張処理（カーネルサイズ・膨張処理回数）の設定
column_horizontal_kernel_hight = 2 # カーネルの縦の高さ
column_horizontal_kernel_width = 5  # カーネルの横の幅
column_horizontal_iterations = 6    # 膨張処理回数

#【縦書き】列領域の検出（列検出）のための膨張処理（カーネルサイズ・膨張処理回数）の設定
row_vertical_kernel_hight = 5  # カーネルの縦の高さ
row_vertical_kernel_width = 3  # カーネルの横の幅
row_vertical_iterations = 6    # 膨張処理回数


#【横書き】個別の文字の検出（文字検出）のための膨張処理（カーネルサイズ・膨張処理回数）の設定
character_horizontal_kernel_hight = 6  # カーネルの縦の高さ
character_horizontal_kernel_width = 3  # カーネルの横の幅
character_horizontal_iterations = 2    # 膨張処理回数

#【縦書き】個別の文字の検出（文字検出）のための膨張処理（カーネルサイズ・膨張処理回数）の設定
character_vertical_kernel_hight = 3  # カーネルの縦の高さ
character_vertical_kernel_width = 5  # カーネルの横の幅
character_vertical_iterations = 2    # 膨張処理回数


# 輪郭のカット設定
# OCRしたい画像に合わせて微調整が必要（画像の大きさが影響します）
# ブロック検出：文字領域検出した輪郭の「横幅」が、以下の範囲なら輪郭を残す
block_horizontal_height_minimum = 5  # 最小値（ピクセル）
block_horizontal_height_max = 1000   # 最大値（ピクセル）

# ブロック検出：文字領域検出した輪郭の「縦の高さ」が、以下の範囲なら輪郭を残す
block_vertical_height_minimum = 5  # 最小値（ピクセル）
block_vertical_height_max = 1000   # 最大値（ピクセル）


# 行検出：文字領域検出した輪郭の「横幅」が、以下の範囲なら輪郭を残す
row_column_horizontal_height_minimum = 5  # 最小値（ピクセル）
row_column_horizontal_height_max = 1000   # 最大値（ピクセル）

# 列検出：文字領域検出した輪郭の「縦の高さ」が、以下の範囲なら輪郭を残す
row_column_vertical_height_minimum = 5  # 最小値（ピクセル）
row_column_vertical_height_max = 1000   # 最大値（ピクセル）


# 個別の文字領域検出した輪郭の「横幅」が、以下の範囲なら輪郭を残す
character_text_detection_horizontal_height_minimum = 5  # 最小値（ピクセル）
character_text_detection_horizontal_height_max = 300    # 最大値（ピクセル）

# 個別の文字領域検出した輪郭の「縦の高さ」が、以下の範囲なら輪郭を残す
character_text_detection_vertical_height_minimum = 10  # 最小値（ピクセル）
character_text_detection_vertical_height_max = 300     # 最大値（ピクセル）





####  ４．直線の検出と除去 ####
# 元の画像から直線を検出し、直線を除去します。
# 「line_cut_元の画像のファイル名.png」を作成します

def line_cut(OCR_input_file):

  # 画像の読み込み
  img = cv2.imread(OCR_input_file)

  print('【直線を検出中・・・】直線検出する画像')
  # 画像の表示
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.show()

  # モノクロ・グレースケール画像へ変換（2値化前の画像処理）
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 2値化（Binarization）：白（1）黒（0）のシンプルな2値画像に変換
  retval, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  #plt.imshow(img_binary)
  print('【直線を検出中・・・】2値化処理画像 - Binarization')
  plt.imshow(cv2.cvtColor(img_binary, cv2.COLOR_BGR2RGB))
  plt.show()

  # 2値化画像で行う
  # rho：画素単位で計算
  # theta：ラジアン単位で計算
  # threshold：直線とみなされるのに必要な最低限の点の数を意味するしきい値。
  # 確率的ハフ変換：
  # minLineLength：検出する直線の最小の長さを表します。この値より短い線分は検出されません
  # maxLineGap：二つの点を一つの直線とみなす時に許容される最大の長さを表します
  # この値より小さい二つの点は一つの直線とみなされます
  # 必要に応じてパラメータの数値を変更してください
  lines = cv2.HoughLinesP(img_binary, rho=1, theta=np.pi/360, threshold=15, minLineLength=55, maxLineGap=5.4)

  if lines is None:
    print('\n【直線の検出結果】')
    print('　直線は検出されませんでした。')
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    cv2.imwrite(f'line_cut_{file_name}.png', img)
  else:
    print('\n【直線の検出結果】')
    print('　直線が検出されました。検出した直線を削除します。')
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 検出した直線に赤線を引く
        red_lines_img = cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 3)
    print('\n【直線検出部位の視覚化】')
    print('　赤色部分が検出できた直線。')
    # 画像の表示（直線を検出した画像）
    plt.imshow(cv2.cvtColor(red_lines_img, cv2.COLOR_BGR2RGB))
    plt.show()

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 検出した直線を消す（白で線を引く）：2値化した際に黒で表示される
        no_lines_img = cv2.line(img, (x1,y1), (x2,y2), (255,255,255), 3)

        # 直線を除去した画像を元のファイル名の頭に「line_cut_」をつけて保存。「0」を指定でファイル名を取得
        file_name = os.path.splitext(os.path.basename(input_file))[0]
        cv2.imwrite(f'line_cut_{file_name}.png', no_lines_img)
    print('\n【直線検出部位の削除結果：元の画像から削除】')
    print('　白色部分が検出した直線を消した場所（背景が白の場合は区別できません）。')
    # 画像の表示（直線を削除した画像）
    plt.imshow(cv2.cvtColor(no_lines_img, cv2.COLOR_BGR2RGB))
    plt.show()

    line_cut_input_file = f'line_cut_{file_name}.png'
    img = cv2.imread(line_cut_input_file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2値化（Binarization）：白（1）黒（0）のシンプルな2値画像に変換
    retval, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print('\n【直線検出部位の削除結果：2値化処理画像 - Binarization】')
    print('　直線を除去した結果。')
    # 画像の表示（直線を削除した2値画像）
    plt.imshow(cv2.cvtColor(img_binary, cv2.COLOR_BGR2RGB))
    plt.show()





## 直線の検出と除去の関数（line_cut）の実行
line_cut(input_file)






####  ５．大まかな文字領域の検出（ブロック検出）＋ ノイズ除去 ####
# 「line_cut_元の画像のファイル名.png」（直線除去画像）からノイズを除去し、ブロック検出をおこないます
# 「block_ROI_img〜.png」（ブロック検出画像）を作成します

# 文字領域を検出・抽出する処理
def block_contours (OCR_input_file):

  # 画像の読み込み
  img = cv2.imread(OCR_input_file)

  # ノイズ除去（Denoising・Noise Reduction）：メディアンフィルタの利用
  #「3」：カーネルサイズ（1・3・5・7など）を大きくすると、点々などのノイズをより消せるが、ぼやける
  # ここのノイズ除去処理により画像がぼやけるので、ここの記述「img = cv2.medianBlur(img, 3)」をなくして
  # 事前に画像処理ソフトなどでノイズを除去した方がいいのかもしれません
  img = cv2.medianBlur(img, 3)

  # モノクロ・グレースケール画像へ変換（2値化前の画像処理）
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 2値化（Binarization）：白（1）黒（0）のシンプルな2値画像に変換
  retval, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

  # 白部分の膨張処理（Dilation）：モルフォロジー変換 - 2値画像を対象
  if OCR_mode == 0: # 横書きの場合
    kernel = np.ones((block_horizontal_kernel_hight, block_horizontal_kernel_width),np.uint8) # カーネル（構造的要素）：全要素の値が1の縦横が任意のピクセルのカーネル
    img_dilation = cv2.dilate(img_binary,kernel,iterations = block_horizontal_iterations) #「iterations=」繰り返し膨張処理を行う回数を指定
  elif OCR_mode == 1: # 縦書きの場合
    kernel = np.ones((block_vertical_kernel_hight, block_vertical_kernel_width),np.uint8) # カーネル（構造的要素）：全要素の値が1の縦横が任意のピクセルのカーネル
    img_dilation = cv2.dilate(img_binary,kernel,iterations = block_vertical_iterations) #「iterations=」繰り返し膨張処理を行う回数を指定

  # 解説用のコメント（2値化）
  print('\n【2値化処理画像 - Binarization】')
  print('  画像の2値化と白部分の膨張を工夫することで、大まかな文字領域の検出（ブロック検出）をしています。')
  print('  この段階で、文字が「白」として処理できていないと輪郭の検出がしにくいようでした。')

  # 膨張処理後の2値化画像の表示
  plt.imshow(cv2.cvtColor(img_dilation, cv2.COLOR_BGR2RGB))
  plt.show()


  # 輪郭の検出
  #「findContours」の返り値「cnts（contours）」は輪郭毎の座標組・「hierarchy」は輪郭の階層構造
  #「cv2.RETR_EXTERNAL」：最も外側の輪郭を返す
  #「cv2.CHAIN_APPROX_SIMPLE」：輪郭上の全点の情報を保持しない。輪郭の情報を圧縮
  cnts, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  if OCR_mode == 0: # 横書きの場合
    cnts, hierarchy = contours.sort_contours(cnts, method='top-to-bottom') # 上から下に並び替え
  elif OCR_mode == 1: # 縦書きの場合
    cnts, hierarchy = contours.sort_contours(cnts, method='right-to-left') # 右から左に並び替え

  # ROI（Region of Interest：興味領域・対象領域）抽出用の初期設定
  block_ROI_index = 0

  # 抽出した輪郭を「x, y, w（横の幅）, h（縦の高さ）」の単純なリストに変換
  result = []
  for contour in cnts:  
    x, y, w, h = cv2.boundingRect(contour) # 外接矩形の左上の位置は(x,y)，横の幅と縦の高さは(w,h)
      
    # 大きすぎる小さすぎる領域を除去。処理する画像サイズに合わせて微調整が必要
    if not block_vertical_height_minimum < w < block_vertical_height_max:
      continue
    if not block_horizontal_height_minimum < h < block_horizontal_height_max: #輪郭の描画は画像サイズを超えることもあるようでした。
      continue

    # ROI抽出：画像の切り抜きと保存 
    block_ROI = img[y:y+h, x:x+w]   
    cv2.imwrite('block_ROI_img{}.png'.format(block_ROI_index), block_ROI)
    block_ROI_index += 1
    #resultに要素を追加
    result.append([x, y, w, h])


  # 画面に矩形の輪郭を描画 （描画機能）
  for x, y, w, h in result:
      cv2.rectangle(img, (x, y), (x+w, y+h), (100, 255, 100), 3)  # 色の指定はRGB(100, 255, 100)。「3」は 太さ。数字を大きくすると太い輪郭が描画される。


  # 解説用のコメント（文字領域の輪郭検出・抽出）
  if OCR_mode == 0: # 横書きの場合
    print('\n【横書きの文字領域の輪郭検出・抽出結果 - Text detection・Contours】')
  elif OCR_mode == 1:  #縦書きの場合
    print('\n【縦書きの文字領域の輪郭検出・抽出結果 - Text detection・Contours】')
  print('  枠が大きすぎる場合・小さすぎる場合には輪郭を除去しています。画像によって微調整する必要があります。')

  # 文字領域の輪郭検出・抽出結果の表示
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  # plt.savefig('block_text-detection.png', dpi=300)  # 抽出結果の画像ファイルも保存しています。「dpi」は何も指定しないと dpi=72
  plt.show()



# 元の画像ファイル名に「line_cut_」を追加
# 直線の検出と除去の関数（line_cut）を実行後に出力される画像をプログラムで使うための処理
file_name = os.path.splitext(os.path.basename(input_file))[0]
line_cut_input_file = f'line_cut_{file_name}.png'



## 大まかな文字領域の検出（ブロック検出）の関数（block_contours）の実行
block_contours (line_cut_input_file)





#### ６．角度補正 ####
# 「block_ROI_img〜.png」（ブロック検出画像）を、ブロックごとに角度補正します
# 「rotate_元の画像のファイル名.png」（角度補正した画像）を作成します

def rotate_program(OCR_input_file):

  # ブロブ処理（粒子解析・ラベリング）
  # 画像内のブロブを検出するために、グレースケールと適応しきい値に変換してバイナリ画像を取得します
  # ブロブ（塊・連結領域）とは、似た特徴を持った画像内の領域を意味
  # ブロブ検出は類似した色の連結領域（ブロブ）を識別するために画像を分析
  # ブロブ解析（中心座標やサイズなど取得）
  # 画像をラベリング処理し、ラベル付けされた領域の特徴を解析することをブロブ解析
  block_img = cv2.imread(OCR_input_file)
  print('【元の画像】')
  plt.imshow(cv2.cvtColor(block_img, cv2.COLOR_BGR2RGB))
  plt.show()

  # モノクロ・グレースケール画像へ変換（2値化前の画像処理）
  block_img_gray = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY)

  # 2値化（Binarization）：白（1）黒（0）のシンプルな2値画像に変換
  retval, block_img_binary = cv2.threshold(block_img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

  print('【2値化画像】')
  plt.imshow(cv2.cvtColor(block_img_binary, cv2.COLOR_BGR2RGB))
  plt.show()

  # 輪郭を見つける
  cnts, hierarchy = cv2.findContours(block_img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
  # 輪郭の選択
  # 面積が小さい輪郭削除 「5」のところの数値より小さい輪郭は削除
  cnts = list(filter(lambda cnts: 5 < cv2.contourArea(cnts), cnts))


  # 入力画像のMat（画像用の行列）のコピー
  block_blob_cnts = np.copy(block_img)

  # 輪郭を描画した配列を作成
  cv2.drawContours(block_blob_cnts, cnts, -1, (255,0,0))

  # 外接矩形
  # 入力画像のMat（画像用の行列）のコピー
  block_bounding_img = np.copy(block_img)

  # 画面に矩形の輪郭を描画 
  for contour in cnts:
      x, y, w, h = cv2.boundingRect(contour)
      cv2.rectangle(block_bounding_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

  print('【輪郭を描画した画像】')
  # 輪郭の個数を出力
  print('　ブロブの数: %d' % len(cnts))
  plt.imshow(cv2.cvtColor(block_bounding_img, cv2.COLOR_BGR2RGB))
  plt.show()

  # ブロブの角度からcv2.minAreaRect()で、スキュー角度（傾斜：skew angle）を計算
  # スキュー角度：水平および垂直の配置に画像イメージを返すのに必要な回転の量
  # np.where(block_img_binary > 0)：0より大きい全てのピクセル値の座標を取得
  # np.column_stack：配列を列方向に積み重ねる
  coordinates= np.column_stack(np.where(block_img_binary > 0))

  # 全ての座標の最小回転境界ボックスを計算
  # 長方形が、xy軸と平行な状態の時に「-90」を返す。時計回りに回転させていくと「0」にむかって値が増えていく
  # 長方形は、90度回転すると、xy軸と平行な状態になるため、「-90」に戻る
  angle = cv2.minAreaRect(coordinates)[-1]
  if angle < -45:
      angle = -(90 + angle)
  else:
      angle = -angle

  # アフィン変換を適用してスキュー角度（傾斜：skew angle）を修正
  # 水平になるように回転
  (h, w) = block_img.shape[:2]
  center = (w // 2, h // 2)
  # 回転のための変換行列の生成 
  # cv2.getRotationMatrix2D（入力画像の回転中心, 回転角度 単位は度- 正の値：反時計回り, 等方性スケーリング係数 - 拡大縮小の倍率）
  rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

  # v2.warpAffine（元の画像, cv2.getRotationMatrix2Dで生成した2*3の変換行列, 出力する画像サイズ(縦の高さ, 横の幅)）
  block_rotate_img = cv2.warpAffine(block_img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

  #  元の画像を傾き角度補正（回転）した画像を表示
  print('【角度補正した画像】')
  print('　補正角度：', angle, '度')
  plt.imshow(cv2.cvtColor(block_rotate_img, cv2.COLOR_BGR2RGB))
  plt.show()

  # 角度補正した画像を「rotate_元の画像のファイル名.png」で保存。「0」を指定でファイル名を取得
  file_name = os.path.splitext(os.path.basename(OCR_input_file))[0]
  cv2.imwrite(f'rotate_{file_name}.png', block_rotate_img)




## 角度補正のための関数（rotate_program）の実行


#「block_ROI_img〜.png」（大まかな文字領域の検出：ブロック検出画像）という名前のファイルの取得
file_list = glob.glob('block_ROI_img*png')

#「block_ROI_img〜.png」（〜の部分に0や1などの数字が入る）という名前のファイルを0から順番に並び替え
print(natsorted(file_list))

#「block_ROI_img〜.png」のファイルを順番に角度補正のための関数（rotate_program）に入れる
for file in natsorted(file_list):
  
  rotate_program(file)






#### ７．横書き・縦書きの行・列領域の検出（行と列の検出） #### 
# 「rotate_元の画像のファイル名.png」（角度補正した画像）から行と列の検出をします
# 「row_column_ROI_img〜.png」（行や列を検出した画像）を作成します

def text_row_column_detection (OCR_input_file):
  # 画像の読み込み
  row_column_img = cv2.imread(OCR_input_file)
  
  # モノクロ・グレースケール画像へ変換（2値化前の画像処理）
  row_column_img_gray = cv2.cvtColor(row_column_img, cv2.COLOR_BGR2GRAY)

  # 2値化（Binarization）：白（1）黒（0）のシンプルな2値画像に変換
  retval, row_column_img_binary = cv2.threshold(row_column_img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

  # 白部分の膨張処理（Dilation）：モルフォロジー変換 - 2値画像を対象
  if OCR_mode == 0: # 横書きの場合
    kernel = np.ones((column_horizontal_kernel_hight, column_horizontal_kernel_width),np.uint8) # カーネル（構造的要素）：全要素の値が1の縦横が任意のピクセルのカーネル
    row_column_img_dilation = cv2.dilate(row_column_img_binary,kernel,iterations = column_horizontal_iterations) #「iterations=」繰り返し膨張処理を行う回数を指定  
    print('\n【各行の2値化処理画像 - Binarization】') 
  elif OCR_mode == 1:  #縦書きの場合
    kernel = np.ones((row_vertical_kernel_hight, row_vertical_kernel_width),np.uint8) # カーネル（構造的要素）：全要素の値が1の縦横が任意のピクセルのカーネル
    row_column_img_dilation = cv2.dilate(row_column_img_binary,kernel,iterations = row_vertical_iterations) #「iterations=」繰り返し膨張処理を行う回数を指定  
    print('\n【各列の2値化処理画像 - Binarization】')
  print('  画像の2値化と白部分の膨張を工夫することで、文字と文字の繋がりを検出しています。')
  print('  画像のテキストの文字の「太さ」「行間」「文字間隔」によっては、画像のリサイズの微調整や膨張処理の微調整が必要です。')


  # 膨張処理後の2値化画像の表示
  plt.imshow(cv2.cvtColor(row_column_img_dilation, cv2.COLOR_BGR2RGB))
  plt.show()

  # 輪郭の検出
  #「findContours」の返り値「contours」は輪郭毎の座標組・「hierarchy」は輪郭の階層構造
  #「cv2.RETR_EXTERNAL」：最も外側の輪郭を返す
  #「cv2.CHAIN_APPROX_SIMPLE」：輪郭上の全点の情報を保持しない。輪郭の情報を圧縮
  cnts, hierarchy = cv2.findContours(row_column_img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  if OCR_mode == 0: # 横書きの場合
      cnts, hierarchy = contours.sort_contours(cnts, method='top-to-bottom') # 上から下に並び替え
  elif OCR_mode == 1: # 縦書きの場合
      cnts, hierarchy = contours.sort_contours(cnts, method='right-to-left') # 右から左に並び替え

  # 抽出した輪郭を「x, y, w（横の幅）, h（縦の高さ）」の単純なリストに変換
  for row_column_contour in cnts:  
    x, y, w, h = cv2.boundingRect(row_column_contour) # 外接矩形の左上の位置は(x,y)，横の幅と縦の高さは(w,h)
    
    # 大きすぎる小さすぎる領域を除去。処理する画像サイズに合わせて微調整が必要
    if not row_column_vertical_height_minimum < w < row_column_vertical_height_max:
      continue
    if not row_column_horizontal_height_minimum < h < row_column_horizontal_height_max: #輪郭の描画は画像サイズを超えることもあるようでした。
      continue

  # 抽出した輪郭を「x, y, w（横の幅）, h（縦の高さ）」の単純なリストに変換
  row_column_result = []
  for row_column_contour in cnts:  
    x, y, w, h = cv2.boundingRect(row_column_contour) # 外接矩形の左上の位置は(x,y)，横の幅と縦の高さは(w,h)
    
    # 大きすぎる小さすぎる領域を除去。処理する画像サイズに合わせて微調整が必要
    if not row_column_vertical_height_minimum < w < row_column_vertical_height_max:
      continue
    if not row_column_horizontal_height_minimum < h < row_column_horizontal_height_max: #輪郭の描画は画像サイズを超えることもあるようでした。
      continue

    #resultに要素を追加
    row_column_result.append([x, y, w, h])

  # 画面に矩形の輪郭を描画 （描画機能：for〜cv2.rectangleまでの2行をコメントアウト、または削除すると輪郭の描画を無効にできます）
  for x, y, w, h in row_column_result:
      cv2.rectangle(row_column_img, (x, y), (x+w, y+h), (100, 255, 100), 2)  # 色の指定はRGB(100, 255, 100)。「2」は 太さ。数字を大きくすると太い輪郭が描画される。


  # 解説用のコメント（輪郭検出・抽出）
  if OCR_mode == 0: # 横書きの場合
    print('\n【各行の文字の輪郭検出・抽出結果 - Text recognition・Contours】')
  elif OCR_mode == 1:  #縦書きの場合
    print('\n【各列の文字の輪郭検出・抽出結果 - Text recognition・Contours】')
  print('  枠が大きすぎる場合・小さすぎる場合には輪郭を除去しています。画像によって微調整する必要があります。')

  # 輪郭検出・抽出結果の表示
  plt.imshow(cv2.cvtColor(row_column_img, cv2.COLOR_BGR2RGB))
  plt.show()




##「row_column_ROI_img〜.png」（行や列を検出した画像）の画像を保存するための処理


#「rotate_block_ROI_img〜.png」（角度補正後のブロック検出画像）という名前のファイルの取得
file_list = glob.glob('rotate_block_ROI_img*png')

#「rotate_block_ROI_img〜.png」（〜の部分に0や1などの数字が入る）という名前のファイルを0から順番に並び替え
print(natsorted(file_list))

# ROI（Region of Interest：興味領域・対象領域）抽出用の初期設定
row_column_ROI_index = 0

#「rotate_block_ROI_img〜.png」のファイルを順番に処理する
for file in natsorted(file_list):
  row_column_img = cv2.imread(file)

  # モノクロ・グレースケール画像へ変換（2値化前の画像処理）
  row_column_img_gray = cv2.cvtColor(row_column_img, cv2.COLOR_BGR2GRAY)

  # 2値化（Binarization）：白（1）黒（0）のシンプルな2値画像に変換
  retval, row_column_img_binary = cv2.threshold(row_column_img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

  if OCR_mode == 0: # 横書きの場合
    kernel = np.ones((column_horizontal_kernel_hight, column_horizontal_kernel_width),np.uint8) # カーネル（構造的要素）：全要素の値が1の縦横が任意のピクセルのカーネル
    row_column_img_dilation = cv2.dilate(row_column_img_binary,kernel,iterations = column_horizontal_iterations) #「iterations=」繰り返し膨張処理を行う回数を指定  
    print('\n【各行の2値化処理画像 - Binarization】') 
  elif OCR_mode == 1:  #縦書きの場合
    kernel = np.ones((row_vertical_kernel_hight, row_vertical_kernel_width),np.uint8) # カーネル（構造的要素）：全要素の値が1の縦横が任意のピクセルのカーネル
    row_column_img_dilation = cv2.dilate(row_column_img_binary,kernel,iterations = row_vertical_iterations) #「iterations=」繰り返し膨張処理を行う回数を指定  

  cnts, hierarchy = cv2.findContours(row_column_img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if OCR_mode == 0: # 横書きの場合
      cnts, hierarchy = contours.sort_contours(cnts, method='top-to-bottom') # 上から下に並び替え
  elif OCR_mode == 1: # 縦書きの場合
      cnts, hierarchy = contours.sort_contours(cnts, method='right-to-left') # 右から左に並び替え

  # 抽出した輪郭を「x, y, w（横の幅）, h（縦の高さ）」の単純なリストに変換
  for row_column_contour in cnts:  
    x, y, w, h = cv2.boundingRect(row_column_contour) # 外接矩形の左上の位置は(x,y)，横の幅と縦の高さは(w,h)

    # ROI抽出：画像の切り抜きと保存。 
    row_column_ROI = row_column_img[y:y+h, x:x+w]   
    cv2.imwrite('row_column_ROI_img{}.png'.format(row_column_ROI_index), row_column_ROI)

    row_column_ROI_index += 1






#### ７．個別の文字の検出（文字検出）＋ 個別の文字検出枠の描画設定 #### 
# 「row_column_ROI_img〜.png」（行や列を検出した画像）から、行や列ごとに個別の文字を検出します
# 「OCR_img〜.png」（個別文字検出画像）を作成します
# 「個別の文字検出枠の描画設定：画面に矩形の輪郭を描画」のコードをコメントアウトまたは削除すると個別の文字検出枠を無効化できます
# 個別の文字検出枠の有無や太さによってOCR結果の精度に影響がでます。基本的に枠線が無い方が精度は良いです
# 個別の文字検出枠があると、どの程度文字検出ができているか把握しやすいので、お好みに応じて調整してみてください

def find_draw_contours (OCR_input_file):
  # 画像の読み込み
  predict_img = cv2.imread(OCR_input_file)
  
  # モノクロ・グレースケール画像へ変換（2値化前の画像処理）
  predict_img_gray = cv2.cvtColor(predict_img, cv2.COLOR_BGR2GRAY)

  # 2値化（Binarization）：白（1）黒（0）のシンプルな2値画像に変換
  retval, predict_img_binary = cv2.threshold(predict_img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

  # 白部分の膨張処理（Dilation）：モルフォロジー変換 - 2値画像を対象
  if OCR_mode == 0: # 横書きの場合
    kernel = np.ones((character_horizontal_kernel_hight, character_horizontal_kernel_width),np.uint8) # カーネル（構造的要素）：全要素の値が1の縦横が任意のピクセルのカーネル
    predict_img_dilation = cv2.dilate(predict_img_binary,kernel,iterations = character_horizontal_iterations) #「iterations=」繰り返し膨張処理を行う回数を指定
  elif OCR_mode == 1: #縦書きの場合
    kernel = np.ones((character_vertical_kernel_hight, character_vertical_kernel_width),np.uint8) # カーネル（構造的要素）：全要素の値が1の縦横が任意のピクセルのカーネル
    predict_img_dilation = cv2.dilate(predict_img_binary,kernel,iterations = character_vertical_iterations) #「iterations=」繰り返し膨張処理を行う回数を指定

  # 解説用のコメント（2値化）
  if OCR_mode == 0: # 横書きの場合
    print('\n【白部分の膨張処理 - Dilation】各行の2値化処理画像 - Binarization')
  elif OCR_mode == 1:  #縦書きの場合
    print('\n【白部分の膨張処理 - Dilation】各列の2値化処理画像 - Binarization')
  print('  画像の2値化と白部分の膨張を工夫することで、個別の文字を検出しやすいようにしています。')
  print('  画像のテキストの文字の「太さ」「行間」「文字間隔」によっては、画像のリサイズや膨張処理の微調整が必要です。')
  print('  この段階で、文字が他の文字と繋がってしまうと、個別の文字の検出ができなくなります。')

  # 膨張処理後の2値化画像の表示
  plt.imshow(cv2.cvtColor(predict_img_dilation, cv2.COLOR_BGR2RGB))
  plt.show()



  # 輪郭の検出
  #「findContours」の返り値「contours」は輪郭毎の座標組・「hierarchy」は輪郭の階層構造
  #「cv2.RETR_EXTERNAL」：最も外側の輪郭を返す
  #「cv2.CHAIN_APPROX_SIMPLE」：輪郭上の全点の情報を保持しない。輪郭の情報を圧縮
  cnts, hierarchy = cv2.findContours(predict_img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if OCR_mode == 0: # 横書きの場合
      cnts, hierarchy = contours.sort_contours(cnts, method='left-to-right') # 左から右に並び替え
  elif OCR_mode == 1: # 縦書きの場合
      cnts, hierarchy = contours.sort_contours(cnts, method='top-to-bottom') # 上から下に並び替え

  # 画像の上に数字の結果を出すためにここでもROI設定は必要
  # ROI（Region of Interest：興味領域・対象領域）抽出用の初期設定
  OCR_index = 0

  # 抽出した輪郭を「x, y, w（横の幅）, h（縦の高さ）」の単純なリストに変換
  predict_result = []
  for predict_contour in cnts:  
    x, y, w, h = cv2.boundingRect(predict_contour) # 外接矩形の左上の位置は(x,y)，横の幅と縦の高さは(w,h)
    
    # 大きすぎる小さすぎる領域を除去。処理する画像サイズに合わせて微調整が必要
    if not character_text_detection_horizontal_height_minimum < w < character_text_detection_horizontal_height_max:
      continue
    if not character_text_detection_vertical_height_minimum < h < character_text_detection_vertical_height_max: #輪郭の描画は画像サイズを超えることもあるようでした。
      continue

    # ROI抽出：画像の切り抜きと保存。→　再学習用に別プログラムを作成する OCR_index自体は、文字数判断のために必要かもしれない
    predict_ROI = predict_img[y:y+h, x:x+w]   
    OCR_index += 1

    #resultに要素を追加
    predict_result.append([x, y, w, h])

  ## 個別の文字検出枠の描画設定：画面に矩形の輪郭を描画 
  # 描画機能：for〜cv2.rectangleまでの2行をコメントアウト、または削除すると輪郭の描画を無効にできます
  # 行・列ごとのOCR結果の表示の際に影響します（線ありの場合、精度が下がります）。「OCR結果：全文」には影響しません
  for x, y, w, h in predict_result:
     cv2.rectangle(predict_img, (x, y), (x+w, y+h), (100, 255, 100), 1)  # 色の指定はRGB(100, 255, 100)。「1」は 太さ。数字を大きくすると太い輪郭が描画される。

  # 解説用のコメント（輪郭検出・抽出）
  if OCR_mode == 0: # 横書きの場合
    print('\n【各行の文字の輪郭検出・抽出結果 - Text recognition・Contours】')
  elif OCR_mode == 1:  #縦書きの場合
    print('\n【各列の文字の輪郭検出・抽出結果 - Text recognition・Contours】')
  print('  枠が大きすぎる場合・小さすぎる場合には輪郭を除去しています。画像によって微調整する必要があります。')

  # 輪郭検出・抽出結果の表示
  plt.imshow(cv2.cvtColor(predict_img, cv2.COLOR_BGR2RGB))
  plt.show()

  return predict_result, predict_img, OCR_index




##「OCR_img〜.png」（個別文字検出）の画像を保存するための処理


#「row_column_ROI_img〜.png」（行や列を検出した画像）という名前のファイルの取得
file_list = glob.glob('row_column_ROI_img*png')

#「row_column_ROI_img〜.png」（〜の部分に0や1などの数字が入る）という名前のファイルを0から順番に並び替え
print(natsorted(file_list))

# ROI（Region of Interest：興味領域・対象領域）抽出用の初期設定
OCR_index = 0

#「row_column_ROI_img〜.png」のファイルを順番に処理する
for file in natsorted(file_list):
  predict_img = cv2.imread(file)

  # モノクロ・グレースケール画像へ変換（2値化前の画像処理）
  predict_img_gray = cv2.cvtColor(predict_img, cv2.COLOR_BGR2GRAY)

  # 2値化（Binarization）：白（1）黒（0）のシンプルな2値画像に変換
  retval, predict_img_binary = cv2.threshold(predict_img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

  # 白部分の膨張処理（Dilation）：モルフォロジー変換 - 2値画像を対象
  if OCR_mode == 0: # 横書きの場合
    kernel = np.ones((character_horizontal_kernel_hight, character_horizontal_kernel_width),np.uint8) # カーネル（構造的要素）：全要素の値が1の縦横が任意のピクセルのカーネル
    predict_img_dilation = cv2.dilate(predict_img_binary,kernel,iterations = character_horizontal_iterations) #「iterations=」繰り返し膨張処理を行う回数を指定
  elif OCR_mode == 1: #縦書きの場合
    kernel = np.ones((character_vertical_kernel_hight, character_vertical_kernel_width),np.uint8) # カーネル（構造的要素）：全要素の値が1の縦横が任意のピクセルのカーネル
    predict_img_dilation = cv2.dilate(predict_img_binary,kernel,iterations = character_vertical_iterations) #「iterations=」繰り返し膨張処理を行う回数を指定

  cnts, hierarchy = cv2.findContours(predict_img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if OCR_mode == 0: # 横書きの場合
      cnts, hierarchy = contours.sort_contours(cnts, method='left-to-right') # 左から右に並び替え
  elif OCR_mode == 1: # 縦書きの場合
      cnts, hierarchy = contours.sort_contours(cnts, method='top-to-bottom') # 上から下に並び替え

  # 抽出した輪郭を「x, y, w（横の幅）, h（縦の高さ）」の単純なリストに変換
  predict_result = []
  for predict_contour in cnts:  
    x, y, w, h = cv2.boundingRect(predict_contour) # 外接矩形の左上の位置は(x,y)，横の幅と縦の高さは(w,h)
    
    # 大きすぎる小さすぎる領域を除去。処理する画像サイズに合わせて微調整が必要
    if not character_text_detection_horizontal_height_minimum < w < character_text_detection_horizontal_height_max:
      continue
    if not character_text_detection_vertical_height_minimum < h < character_text_detection_vertical_height_max: #輪郭の描画は画像サイズを超えることもあるようでした。
      continue

    # ROI抽出：画像の切り抜きと保存。→　再学習用に別プログラムを作成する OCR_index自体は、文字数判断のために必要かもしれない。
    predict_ROI = predict_img[y:y+h, x:x+w]   
    cv2.imwrite('OCR_img{}.png'.format(OCR_index), predict_ROI)
    OCR_index += 1





#### ８．画像判定のためのプログラム ####
# オリジナルデータセットで学習させた学習済みモデルを使って画像判定
# 「行や列ごとの判定」「全文の判定」ができます

# 画像の各種設定
# 認識できる文字を増やしたい場合は、ここに文字を追加します（事前に、文字を増やした「学習済みモデル」を作成する必要があります）
# ステップ1〜2 で利用した以下のプログラム
# 【日本語の手書き文字画像認識用：Pythonサンプルコード】KerasでCNN機械学習。自作・自前画像のオリジナルデータセットで画像認識入門 のプログラム
# Keras-CNN-Japanese-handwritten-character-text-originaldataset.ipynb
# https://colab.research.google.com/drive/1TEjxN8xZVC0k08WzG_Ie8dyaGloSqJzA?usp=sharing
# 内の「④ 自前画像で判定（手書き日本語画像）」で出力された順番に合わせて文字を追加します
folder=['あ','い','う','え','お','か','が','き','ぎ','く','ぐ','け','げ',
        'こ','ご','さ','ざ','し','じ','す','ず','せ','ぜ','そ','ぞ','た',
        'だ','ち','ぢ','っ','つ','づ','て','で','と','ど','な','に','ぬ',
        'ね','の','は','ば','ぱ','ひ','び','ぴ','ふ','ぶ','ぷ','へ','べ',
        'ぺ','ほ','ぼ','ぽ','ま','み','む','め','も','ゃ','や','ゅ','ゆ',
        'ょ','よ','ら','り','る','れ','ろ','わ','を','ん','一','丁','七',
        '万','三','上','下','不','世','両','中','主','久','乗','九','予',
        '争','事','二','五','交','京','人','仁','今','仏','仕','他','付',
        '代','令','以','仮','件','任','休','会','伝','似','位','低','住',
        '体','何','余','作','使','例','供','依','価','便','係','俗','保',
        '信','修','俵','倉','個','倍','候','借','停','健','側','備','働',
        '像','億','元','兄','先','光','児','党','入','全','八','公','六',
        '共','兵','具','典','兼','内','円','再','写','冬','冷','処','出',
        '刀','分','切','刊','列','初','判','別','利','制','刷','券','則',
        '前','創','力','功','加','助','努','労','効','勇','勉','動','務',
        '勝','勢','勤','勧','包','化','北','区','医','十','千','午','半',
        '卒','協','南','単','博','印','厚','原','厳','去','参','友','反',
        '収','取','受','口','古','句','可','台','史','右','号','司','各',
        '合','同','名','后','向','君','否','告','周','味','命','和','品',
        '員','唱','商','問','善','喜','営','器','四','回','因','団','囲',
        '図','固','国','園','土','圧','在','地','坂','均','型','基','堂',
        '報','場','塩','境','墓','増','士','声','壱','売','変','夏','夕',
        '外','多','夜','大','天','太','夫','央','失','奮','女','妹','妻',
        '姉','始','委','婦','子','字','存','孝','季','学','孫','守','安',
        '完','宗','官','定','実','客','宣','室','宮','害','家','容','宿',
        '寄','富','寒','察','寺','対','専','尊','導','小','少','就','局',
        '居','届','屋','展','属','山','岩','岸','島','川','州','工','左',
        '差','己','市','布','希','師','席','帯','帰','帳','常','幅','平',
        '年','幸','幹','広','序','底','店','府','度','庫','庭','康','延',
        '建','弁','式','弐','引','弟','弱','張','強','当','形','役','往',
        '待','律','後','徒','従','得','復','徳','心','必','志','応','忠',
        '快','念','思','急','性','恩','息','悪','悲','情','想','意','愛',
        '感','態','慣','憲','成','我','戦','戸','所','手','才','打','承',
        '技','投','折','招','拝','拡','拾','持','指','挙','授','採','接',
        '推','提','損','支','改','放','政','故','救','敗','教','散','敬',
        '数','整','敵','文','料','断','新','方','旅','族','旗','日','旧',
        '早','明','易','星','春','昨','昭','是','昼','時','景','晴','暑',
        '暗','暴','曜','曲','書','最','月','有','服','望','朝','期','木',
        '未','末','本','材','村','条','来','東','板','林','果','柱','査',
        '栄','校','株','根','格','案','械','森','植','検','業','極','楽',
        '構','様','標','権','横','橋','機','欠','次','欲','歌','歓','止',
        '正','武','歩','歯','歴','死','残','殺','母','毎','毒','比','毛',
        '氏','民','気','水','氷','永','求','池','決','汽','河','油','治',
        '法','波','注','泳','洋','活','派','流','浅','浴','海','消','液',
        '深','混','清','済','減','温','測','港','湖','湯','満','準','漁',
        '演','漢','潔','火','災','炭','点','無','然','焼','照','熱','燃',
        '燈','父','版','牛','牧','物','特','犬','犯','状','独','率','玉',
        '王','現','球','理','生','産','用','田','由','申','男','町','画',
        '界','畑','留','略','番','異','疑','病','発','登','白','百','的',
        '皇','皮','益','盟','目','直','相','省','県','真','眼','着','知',
        '短','石','研','破','確','示','礼','社','祖','祝','神','票','祭',
        '禁','福','私','秋','科','秒','称','移','程','税','種','穀','積',
        '究','空','立','章','童','競','竹','第','筆','等','答','策','算',
        '管','節','築','米','粉','精','糸','系','紀','約','納','純','紙',
        '級','素','細','終','組','経','結','給','統','絵','絶','絹','続',
        '綿','総','緑','線','編','練','績','織','罪','置','美','群','義',
        '習','老','考','者','耕','耳','聖','聞','職','肉','肥','育','胃',
        '能','脈','腸','臣','臨','自','至','興','舌','舎','航','船','良',
        '色','花','芸','芽','苦','英','茶','草','荷','菜','落','葉','著',
        '蔵','薬','虫','蚕','血','衆','行','術','衛','表','補','製','複',
        '西','要','見','規','視','覚','親','観','角','解','言','計','討',
        '訓','記','設','許','訳','証','評','詞','試','詩','話','認','語',
        '誠','誤','説','読','課','調','談','論','諸','講','謝','識','議',
        '護','谷','豊','象','貝','負','財','貧','貨','責','貯','貴','買',
        '貸','費','貿','賀','賃','資','賛','賞','質','赤','走','起','足',
        '路','身','車','軍','転','軽','輪','輸','辞','農','辺','近','返',
        '述','迷','追','退','送','逆','通','速','造','連','週','進','遊',
        '運','過','道','達','遠','適','選','遺','郡','部','都','配','酒',
        '酸','釈','里','重','野','量','金','鉄','鉱','銀','銅','銭','録',
        '鏡','長','門','開','間','関','防','限','陛','院','除','陸','険',
        '陽','隊','階','際','集','雑','難','雨','雪','雲','電','需','青',
        '静','非','面','革','音','順','預','領','頭','題','額','顔','願',
        '類','風','飛','食','飯','飲','養','館','首','馬','駅','験','高',
        '魚','鳥','鳴','麦','黄','黒','鼻']

# 行や列ごとに個別文字認識をする処理
def cognition(OCR_input_file):
  # 解説用コメント
  if OCR_mode == 0: # 横書きの場合
    print('\n【各行の画像判別結果 - Prediction】デフォルトではグレースケールの画像を判定できます。')
  elif OCR_mode == 1:  #縦書きの場合
     print('\n【各列の画像判別結果 - Prediction】デフォルトではグレースケールの画像を判定できます。')
  print(' 「color_setting = 3」に変更するとカラー版の学習済みモデルにも対応できます。 \n')
  print('  OCR結果（予測結果）： \n')
  # 読み込んだ画像データを予測
  for i, predict_contour in enumerate(cnts):
      x, y, w, h = predict_contour 

      # 画像データを取り出す
      img_extraction = predict_img[y:y+h, x:x+w]

      # データを学習済みモデルのデータに合わせる
      if color_setting == 1:
        gazou = cv2.cvtColor(img_extraction, cv2.COLOR_BGR2GRAY)  # モノクロ・グレースケールの場合
      elif color_setting == 3:
        gazou = cv2.cvtColor(img_extraction, cv2.COLOR_BGR2RGB)   # カラーの場合
      gazou = cv2.resize(gazou, (image_width, image_height))
      suuti = gazou.reshape(image_width, image_height, color_setting).astype('float32')/255 

      # 予測する
      n = model.predict(np.array([suuti]))

    
      # 画面に結果を表示
      plt.subplot(1, OCR_index, i + 1) 
      plt.imshow(cv2.cvtColor(img_extraction, cv2.COLOR_BGR2RGB))
      plt.axis('off')
      plt.title(folder[n.argmax()] )
      print(n.argmax())
      print(folder[n.argmax()] , end='')  #横一列に表示させるため「, end=''」を追加
  plt.show()



#「OCR結果：全文」を表示させる処理
def cognition_text(OCR_input_file):
  # 読み込んだ画像データを予測
  if color_setting == 1: # モノクロ・グレースケールの場合
    gazou = cv2.imread(OCR_input_file, 0)
  elif color_setting == 3:  # カラーの場合
    gazou = cv2.imread(OCR_input_file, 1)

  # データを学習済みモデルのデータに合わせる
  gazou = cv2.resize(gazou, (image_width, image_height))
  if color_setting == 1: # モノクロ・グレースケールの場合
    suuti = gazou.reshape(image_width, image_height, 1).astype('float32')/255 
  elif color_setting == 3:  # カラーの場合
    suuti = gazou.reshape(image_width, image_height, 3).astype('float32')/255 

  # 予測する
  n = model.predict(np.array([suuti]))
  print(folder[n.argmax()] , end='')  #横一列に表示させるため「, end=' '」を追加





#### ９．関数の実行など #### 

## 横書き・縦書きの行・列領域の検出（行と列の検出）する関数（text_row_column_detection）の実行
#「rotate_block_ROI_img〜.png」（角度補正後のブロック検出画像）という名前のファイルの取得
file_list = glob.glob('rotate_block_ROI_img*png')
print(natsorted(file_list))
#「rotate_block_ROI_img〜.png」の画像を0から順番に1つずつ処理
for file in natsorted(file_list):
  text_row_column_detection(file)


## 個別の文字の検出（文字検出）する関数（find_draw_contours）の実行
## 個別の文字認識をする関数（cognition）の実行
#「row_column_ROI_img〜.png」（行や列を検出した画像）という名前のファイルの取得
file_list = glob.glob('row_column_ROI_img*png')
print(natsorted(file_list))
#「row_column_ROI_img〜.png」を0から順番に1つずつ処理
for file in natsorted(file_list):
  cnts, predict_img,OCR_index = find_draw_contours(file) 
  cognition(file)


##「OCR結果：全文」を表示させる関数（cognition_text）の実行
#「OCR_img〜.png」（個別の文字検出画像）という名前のファイルの取得
file_list = glob.glob('OCR_img*png')
print(natsorted(file_list))
print('\n【OCR結果：全文】\n')
#「OCR_img〜.png」の画像を0から順番に1つずつ認識させることで全文結果を表示
for file in natsorted(file_list):
  print('a')
  cognition_text(file)


# 元の画像を表示させる処理など
print('\n\n【OCRした画像】\n')
img = cv2.imread(input_file)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


print('\n　間違えた場合は、間違えた文字の画像（OCR_img〜.png：出力結果は画像の番号順）で再学習してみてください。\n')






#### １０．出力した画像ファイルの削除のためのプログラム ####
# 今回のコードでは、画像を一度作成後に、プログラム内で利用し、最後に作成した画像を全て削除しています
# そのため、次にOCRを実行する画像内の文字検出数などが少ないと、誤作動が起きるリスクがあります
# 画像ファイルを保存した場合には、一度生成された画像を削除してから、新たな画像のOCRを試みてください
# 「個別の文字領域検出した画像（OCR_img*png）の削除」のコードをコメントアウトまたは、削除すると個別の文字検出画像を保存できます


# 直線検出した画像（line_cut_元のファイル名.png）の削除：
# 保存したい場合は、ここをコメントアウト（または削除）すると画像を保存できます
file_name = os.path.splitext(os.path.basename(input_file))[0]
os.remove(f'line_cut_{file_name}.png')


# 文字領域のブロック検出した画像（block_ROI_img〜.png）の削除：
# 保存したい場合は、ここをコメントアウト（または削除）すると画像を保存できます
file_list = glob.glob("block_ROI_img*png")
for file in file_list:
  os.remove(file)


# 角度補正した画像（rotate_block_ROI_img〜.png）の削除：
# 保存したい場合は、ここをコメントアウト（または削除）すると画像を保存できます
file_list = glob.glob('rotate_block_ROI_img*png')
for file in file_list:
  os.remove(file)


# 行と列の文字領域検出した画像（row_column_ROI_img〜.png）の削除：
# 保存したい場合は、ここをコメントアウト（または削除）すると画像を保存できます
file_list = glob.glob('row_column_ROI_img*png')
for file in file_list:
  os.remove(file)


# 個別の文字領域検出した画像（OCR_img*png）の削除：
# 保存したい場合は、ここをコメントアウト（または削除）すると画像を保存できます
# OCRに失敗した個別の文字画像で再学習させることができます
file_list = glob.glob('OCR_img*png')
for file in file_list:
  os.remove(file)
