###########################################################################
# Intellectual Property Notice
###########################################################################
# This IMR algorithm is the subject of a patent application.
#
# Patent pending: JP Patent Application No. 2025-083331
#
# The source code is provided for academic and research purposes only. 
# No license is granted, either expressly or implicitly, to practice any patented invention.
#
# ==========================================================================
#   IMR - estimating variant proportion / 2026-02-01 動作確認済 v.2.6
# ==========================================================================
#    使い方:
#      # フォルダ一括
#      python IMR.py csv_dir/ constellations/ -l lineages.txt -o out_dir
#      # 1ファイル
#      python IMR.py sample_snv constellations/ -l lineages.txt -o out_dir
#
#   constellations/ : [必須] 各変異株系統のSNVリストがあるフォルダ指定。Alcov のconstellationsを利用 https://github.com/Ellmen/alcov
#   lineages.txt : 検出対象とする系統の一覧（各行1系統で記述）。省略した場合には，constellationsフォルダ内のすべての系統を検出対象。
#   out_dir : 解析結果の出力先ディレクトリ。省略した場合には，カレントディレクトリに出力
#
#   [入力ファイル sample_snv（サンプルの観測SNVリスト）]
#   (1) csv ファイルの場合：
#        ・列名は、position, ref, alt, DP, DPalt、として含めてください（大文字小文字も合わせる）。
#        ・コンマ(,)区切りcsvにしてください。
#   (2) VCFファイルの場合：[動作未確認,修正必要かも]
#       ・各サンプルごとに単一サンプルVCFを作成。
#       ・multi-allelicの記述には未対応です。各行 1 alleleの形式としてください。
#       ・複数サンプル処理には、単一サンプルVCFを一つのフォルダーに入れて一括読み込みしてください。複数サンプルを含むVCFファイルにも未対応です。
#
#   [出力ファイル]
#   dir_out/
#   ├─ variant_proportions_matrix.csv (各サンプルの変異株割合）
#   ├─ lineage_snv_pattern.csv        (照合先の変異株系統のSNVパターン行列）
#   ├─ batch_error.csv                (処理エラーとなったサンプルの情報）
#   │
#   ├─ sample_snv_matched/            (各サンプルにおける、変異株と一致した観測SNVの一覧）
#   │  ├─ sample1_snv_matched.csv
#   │  ├─ sample2_snv_matched.csv
#   │  └─ ...
#   └─ lineage_snv_matched/               (各系統ごとの、各サンプルで観測されたSNVのdepthとSNV割合）
#      ├─ lineage1_snv_matched_af.csv     (SNV割合: AF = DPalt/DP)
#      ├─ lineage1_snv_matched_dp.csv     (Depth: DP)
#      ├─ lineage2_snv_matched_af.csv
#      ├─ lineage2_snv_matched_dp.csv
#      └─ ...
# ==========================================================================
from __future__ import annotations
import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

debug=False  # debug用標準出力
MAX_LOOPS_DEFAULT=30 # 行列縮減の最大ループ数（default）
# =============================================================================
# Step 1: sample_snv（主データ）から Mutation 列を作る
# =============================================================================
def read_vcf_depth_minimal(vcf_path: str, sample_name: str | None = None) -> pd.DataFrame:
    """
    VCFファイルから position, ref, alt, DP, DPalt を抽出して返す。
    返り値: df_out
      columns = ["position", "ref", "alt","DP", "DPalt"]
    """
    # --- ヘッダ取得 ---
    columns = None
    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#CHROM"):
                columns = line.strip().lstrip("#").split("\t")
                break
    if columns is None:
        raise ValueError(f"#CHROM header line not found in VCF: {vcf_path}")
    df = pd.read_csv(vcf_path,sep="\t",comment="#",header=None,names=columns)
    # --- フォーマット確認・エラー処理------
    if df.empty:
        # VCFにレコードがない場合でも空DFを返す
        return pd.DataFrame(columns=["position", "ref", "alt", "DP", "DPalt"])
    if "ALT" in df.columns:
        # multi-allelic 検出（ALTにカンマが含まれる）の場合
        if df["ALT"].astype(str).str.contains(",", regex=False, na=False).any():
            raise ValueError(f"Multi-allelic records detected in ALT. Not supported: {vcf_path}")
    # --- INFO から DP= を抽出 ---
    df["DP_info"] = (df["INFO"].str.extract(r"DP=(\d+)").astype("float"))
    # --- FORMAT 展開（単一サンプル想定） ---
    df["DPalt"] = np.nan  
    if "FORMAT" in df.columns:
        if sample_name is None:
            sample_name = df.columns[-1]
        format_keys = df["FORMAT"].str.split(":")
        format_vals = df[sample_name].str.split(":")
        format_df = pd.DataFrame(format_vals.tolist(),columns=format_keys.iloc[0],index=df.index)
        if "DP" in format_df.columns:
            df["DP"] = pd.to_numeric(format_df["DP"], errors="coerce")
        if "AD" in format_df.columns:
            ad = format_df["AD"].str.split(",", expand=True)
            df["DPalt"] = pd.to_numeric(ad[1], errors="coerce")
    # --- DP が FORMAT に無ければ INFO.DP を使う ---
    if "DP" not in df.columns:
        df["DP"] = df["DP_info"]
    # --- 必要列のみ抽出＆名称変更 ---
    df_out = df.rename(columns={"POS": "position","REF": "ref","ALT": "alt"})[["position", "ref", "alt", "DP", "DPalt"]]
    return df_out
    
def concat_mutation_from_snv_sample(snv_sample_csv: str | Path) -> pd.DataFrame:
    """
    入力ファイル（sample_snv）から position, ref, alt, DP, DPalt を読み込み、
    Mutation = ref + position + alt を作成して返す。
    返り値: df_sample_snv
      columns = ["position", "ref", "alt", "Mutation", "DP", "DPalt"]
    """
    snv_sample_csv = Path(snv_sample_csv)
    # ファイル種別判定（.vcf / その他(=.csv)）
    is_vcf = snv_sample_csv.suffix.lower() == ".vcf"
    if is_vcf:
        df = read_vcf_depth_minimal(str(snv_sample_csv))  
    else:
        df = pd.read_csv(snv_sample_csv)    # vcfでない場合は csvとして処理
        
    REQUIRED_COLS = ["position", "ref", "alt", "DP", "DPalt"]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns are missing: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    df_sample_snv = df[REQUIRED_COLS].copy()
    # Mutation = ref + position + alt（例: C28958A）
    df_sample_snv["Mutation"] = (
        df_sample_snv["ref"].astype(str)
        + df_sample_snv["position"].astype(str)
        + df_sample_snv["alt"].astype(str)
    )
    # 列順を明示的に整理（後段処理との整合性を保つ）
    df_sample_snv = df_sample_snv[
        ["position", "ref", "alt", "Mutation", "DP", "DPalt"]
    ]
    return df_sample_snv
    
# =============================================================================
# Step 2: constellations 読み込み（lineages.txt があればそれだけ、無ければ全部）
# =============================================================================
def _read_lineages_txt(lineages_txt: str | Path) -> List[str]:
    p = Path(lineages_txt)
    if not p.exists():
        raise FileNotFoundError(f"lineages.txt not found: {p}")
    lineages: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lineages.append(s)
    if not lineages:
        raise ValueError(f"No lineages found in: {p}")
    return lineages

def load_constellations_sites(
    constellations_dir: str | Path,
    json_glob: str = "*.json",
    lineages_txt: str | Path | None = None,
) -> Dict[str, set[str]]:
    """
    constellations_dir 配下の JSON を読み、
    {variant_name: set_of_sites} を返す。
    lineages_txt が指定された場合:
      - そのファイルに列挙された系統のみを対象にする（存在しない系統は警告して無視）
    lineages_txt が None の場合:
      - constellations_dir 内の JSON をすべて対象にする
    variant_name: ファイル名（拡張子除く）を採用（例: JN.1.json -> "JN.1"）
    sites: JSON の "sites" 配列（例: ["C21T", ..., "C28958A", ...]）
    """
    constellations_dir = Path(constellations_dir)
    if not constellations_dir.exists():
        raise FileNotFoundError(f"constellations_dir not found: {constellations_dir}")
    # フィルタ対象（指定があれば）
    selected: Optional[set[str]] = None
    if lineages_txt is not None:
        selected = set(_read_lineages_txt(lineages_txt))
    out: Dict[str, set[str]] = {}
    for fp in sorted(constellations_dir.glob(json_glob)):
        variant = fp.stem
        # lineages.txt 指定がある場合は対象外をスキップ
        if selected is not None and variant not in selected:
            continue
        with fp.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        sites = obj.get("sites", None)
        if not isinstance(sites, list):
            continue
        out[variant] = {str(x).strip() for x in sites if str(x).strip()}
    if selected is not None:
        # 指定された系統のうち、実際に読み込めたものを確認
        loaded = set(out.keys())
        missing = sorted(selected - loaded)
        if missing:
            print(f"[WARN] These lineages listed in {lineages_txt} were not found/loaded: {missing}")
    if not out:
        raise ValueError(
            f"No constellation JSON files found (or no valid 'sites') in: {constellations_dir}"
        )
    return out

def _sort_mutations(muts: List[str]) -> List[str]:
    """
    例: A23232T, C28958A のような Mutation 文字列を
    「中央の数字（座位）」で昇順ソートする。
    数字が取れないものは末尾へ。
    """
    def _key(m: str):
        s = str(m)
        mnum = re.search(r"(\d+)", s)  # 文字列中の最初の数字列（座位）を取得
        if mnum:
            pos = int(mnum.group(1))
            return (0, pos, s)        # (正常, 座位, 文字列) で安定ソート
        else:
            return (1, 10**12, s)     # 数字なしは最後へ
    return sorted(muts, key=_key)

def constellations_to_wide_df(constellations_sites: Dict[str, set[str]]) -> pd.DataFrame:
    """
    constellations_sites（{lineage: set(Mutation)}）から
    wide形式（Mutation × lineage の 0/1 行列）を作成し、
    Mutation の position（数値）で昇順ソートする。
    """
    # 全 Mutation のユニーク集合
    all_mutations = _sort_mutations(list({m for muts in constellations_sites.values() for m in muts}))
    df = pd.DataFrame({"Mutation": list(all_mutations)})
    # 系統列を付与
    for lineage, muts in constellations_sites.items():
        df[lineage] = df["Mutation"].isin(muts).astype(int)
    return df
    
def match_mutations_to_constellations(
    df_sample_snv: pd.DataFrame,
    constellations_sites: Dict[str, set[str]],
    mutation_col: str = "Mutation",
) -> pd.DataFrame:
    """
    df_sample_snv の Mutation が各 variant の sites に含まれるかを 0/1 で付与する。
    出力列（要望）:
      ["Mutation", <各系統列...>, "SUM", "DP", "DPalt"]
    ※ position/ref/alt は削除
    ※ Mutation は残して第1列
    """
    if mutation_col not in df_sample_snv.columns:
        raise ValueError(f"'{mutation_col}' column not found in df_sample_snv")
    for col in ["DP", "DPalt"]:
        if col not in df_sample_snv.columns:
            raise ValueError(f"'{col}' column not found in df_sample_snv")
    # Mutation は str 化（NaN対策）
    mutations = df_sample_snv[mutation_col].astype(str)
    # 系統列を作成
    df_flags = pd.DataFrame({"Mutation": mutations})
    lineage_cols: List[str] = []
    for lineage, sites in constellations_sites.items():
        df_flags[lineage] = mutations.isin(sites).astype(int)
        lineage_cols.append(lineage)
    # SUM（何系統に一致したか）
    df_flags["SUM"] = df_flags[lineage_cols].sum(axis=1)
    # DP/DPalt 付与（元データの順序を維持して横付け）
    df_flags["DP"] = df_sample_snv["DP"].to_numpy()
    df_flags["DPalt"] = df_sample_snv["DPalt"].to_numpy()
    # 列順
    df_flags = df_flags[["Mutation", *lineage_cols, "SUM", "DP", "DPalt"]]
    # SUM=0（どの系統にも一致しない行）を削除
    before = len(df_flags)
    df_flags = df_flags[df_flags["SUM"] != 0].reset_index(drop=True)
    after = len(df_flags)
    if after < before:
        print(f"[STEP2] drop SUM==0 rows: {before} -> {after}")
    # SUM 降順（VBAに合わせるならここでソート）
    df_flags = df_flags.sort_values("SUM", ascending=False).reset_index(drop=True)
    return df_flags

# =============================================================================
# 共通: 系統列の抽出
# =============================================================================
def get_lineage_columns(df: pd.DataFrame) -> List[str]:
    """
    0/1 の「系統列」を抽出する。
    ルール: 既知の非系統列を除外し、残りを系統列とみなす。
    """
    non_lineage = {"Mutation", "SUM", "DP", "DPalt", "AF"}
    return [c for c in df.columns if c not in non_lineage]
    
# =============================================================================
# Step 3: 同一列（0/1列が完全一致）の判定 & 統合
# =============================================================================
def has_identical_lineage_columns(df: pd.DataFrame, lineage_cols: Sequence[str]) -> bool:
    """
    0/1系統列に「完全一致列」が存在するか判定
    """
    if len(lineage_cols) <= 1:
        return False
    mat = df[list(lineage_cols)].to_numpy()
    # 列ごとにタプル化して比較（行数が多いと重いが、ここでは明確さ優先）
    seen = {}
    for j, col in enumerate(lineage_cols):
        key = tuple(mat[:, j].tolist())
        if key in seen:
            return True
        seen[key] = col
    return False

def consolidate_identical_lineage_columns(df: pd.DataFrame, lineage_cols: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    完全一致する系統列を統合し、列名を "A / B / C" のように結合する。
    返り値:
      - df_after_consolidation
      - mapping: 元列名 -> 統合後列名（追跡用）
    """
    mat = df[list(lineage_cols)].to_numpy()
    key_to_newname: Dict[Tuple[int, ...], str] = {}
    key_to_cols: Dict[Tuple[int, ...], List[str]] = {}
    for j, col in enumerate(lineage_cols):
        key = tuple(mat[:, j].tolist())
        key_to_cols.setdefault(key, []).append(col)
    # 新しい列名（同一列はまとめる）
    new_cols: List[str] = []
    for key, cols in key_to_cols.items():
        merged = " / ".join(cols) if len(cols) > 1 else cols[0]
        key_to_newname[key] = merged
        new_cols.append(merged)
    # 新しい行列（代表列を1つ採用）
    new_mat = []
    for key, cols in key_to_cols.items():
        rep = cols[0]
        new_mat.append(df[rep].to_numpy())
    new_mat = np.vstack(new_mat).T  # shape: (n_rows, n_new_cols)
    df_out = df.copy()
    # 既存の系統列を落として、新列を追加
    df_out = df_out.drop(columns=list(lineage_cols))
    # 新列を DataFrame で追加（順序保持）
    df_new = pd.DataFrame(new_mat, columns=new_cols, index=df_out.index)
    # Mutation を先頭に置きたいので、いったん結合して後で並べ替え
    df_out = pd.concat([df_out[["Mutation"]], df_new, df_out.drop(columns=["Mutation"])], axis=1)
    # mapping（元→統合後）
    mapping: Dict[str, str] = {}
    for key, cols in key_to_cols.items():
        merged = key_to_newname[key]
        for c in cols:
            mapping[c] = merged
    # SUM を更新（系統列数が変わるので再計算）
    new_lineage_cols = get_lineage_columns(df_out)
    df_out["SUM"] = df_out[new_lineage_cols].sum(axis=1)
    # 列順を整える（Mutation, 系統..., SUM, DP, DPalt）
    tail = [c for c in ["SUM", "DP", "DPalt"] if c in df_out.columns]
    df_out = df_out[["Mutation", *new_lineage_cols, *tail]]
    return df_out, mapping

# =============================================================================
# Step 4: 同一パターン集計（depth付き）
# =============================================================================
def summarize_patterns_with_depth(df_matched: pd.DataFrame) -> pd.DataFrame:
    """
    系統0/1パターン（lineage_cols）だけで集約し、
    DP / DPalt を合算して AF を再計算する。
    Mutation はグループ内の一覧として "Mutation" 列に連結して残す。
    """
    lineage_cols = get_lineage_columns(df_matched)
    if not lineage_cols:
        raise ValueError("No lineage columns found to summarize.")
    df = df_matched.copy()
    # 重要：グループキーは lineage_cols のみ（Mutation は入れない）
    group_cols = list(lineage_cols)
    agg = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            Mutation=("Mutation", lambda s: ";".join(map(str, s))),  # まとめて残す
            DP=("DP", "sum"),
            DPalt=("DPalt", "sum"),
        )
    )
    # AF を再計算
    agg["AF"] = agg["DPalt"] / agg["DP"].replace(0, np.nan)
    # 列順：Mutation を先頭に
    agg = agg[["Mutation", *lineage_cols, "DP", "DPalt", "AF"]]
    return agg

def has_identical_patterns(df: pd.DataFrame, lineage_cols: Sequence[str]) -> bool:
    """
    系統列の「行パターン」が同一のものがあるか判定
    （ここでは、系統列だけで完全一致する行が2つ以上あるか）
    """
    if not lineage_cols:
        return False
    sub = df[list(lineage_cols)]
    return sub.duplicated().any()


# =============================================================================
# Step 5: MatrixE の作成・正方化（DP_sum が小さい行から削る）
# =============================================================================
@dataclass
class SquareResult:
    df_square: pd.DataFrame
    matrixE_square_df: pd.DataFrame
    R: pd.Series  # AF_sum
    lineage_cols: List[str]

def build_matrixE_and_square(
    df_summary: pd.DataFrame,
    drop_by: str = "DP",
) -> SquareResult:
    lineage_cols = get_lineage_columns(df_summary)
    if not lineage_cols:
        raise ValueError("No lineage columns found for matrixE.")
    n_lineages = len(lineage_cols)
    df_sq = df_summary.copy()
    # 行数を系統数に合わせる（DP が小さい行から削除）
    if len(df_sq) > n_lineages:
        df_sq = df_sq.sort_values(drop_by, ascending=True).reset_index(drop=True)
        df_sq = df_sq.iloc[(len(df_sq) - n_lineages):].reset_index(drop=True)
    elif len(df_sq) < n_lineages:
        raise ValueError(
            f"Not enough rows to make matrix square: rows={len(df_sq)} < lineages={n_lineages}"
        )
    matrixE_square_df = df_sq[lineage_cols].astype(float).copy()
    if "AF" not in df_sq.columns:
        raise ValueError("AF not found in df_summary")
    R = df_sq["AF"].astype(float).copy()
    return SquareResult(
        df_square=df_sq,
        matrixE_square_df=matrixE_square_df,
        R=R,
        lineage_cols=lineage_cols,
    )

def try_inverse(matrixE_square_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    逆行列が計算できるなら DataFrame を返し、無理なら None。
    """
    E = matrixE_square_df.to_numpy(dtype=float)
    # 正方チェック
    if E.shape[0] != E.shape[1]:
        return None
    # 特異（det=0）チェック：rankで判定（数値誤差に強い）
    rank = np.linalg.matrix_rank(E)
    if rank < E.shape[0]:
        return None
    E_inv = np.linalg.inv(E)
    return pd.DataFrame(E_inv, index=matrixE_square_df.columns, columns=matrixE_square_df.index)

# =============================================================================
# 最終: X = E_inv @ R（系統割合）を算出し、ファイル名に入力CSV名を含めて出力
# =============================================================================
def estimate_variant_proportions(
    E_inv: pd.DataFrame,
    R: pd.Series,
) -> pd.DataFrame:
    """
    X = E_inv @ R を計算して、系統別割合（正規化）を返す。
    """
    # E_inv: index=lineage, columns=row_id(=df_squareの行 index)
    # R: index=df_squareの行 index に対応（順序は df_square の順）
    r = R.to_numpy(dtype=float).reshape(-1, 1)
    X = E_inv.to_numpy(dtype=float) @ r
    df_X = pd.DataFrame(
        {"Variant_proportion": X.flatten()},
        index=E_inv.index,
    )
    # 負値は0に（数値誤差対策）
    df_X["Variant_proportion"] = df_X["Variant_proportion"].clip(lower=0)
    # 正規化（合計1）
    total = float(df_X["Variant_proportion"].sum())
    if total > 1:
        df_X["Variant_proportion"] /= total; total=1.0
    else:
        total = min(total, 1.0)
    # ② 未同定分（residue）を計算
    residue = max(0.0, 1.0 - total)
    # lineage を列に
    df_X = df_X.reset_index().rename(columns={"index": "lineage"})
    # unidentified_residue を追加
    df_X = pd.concat([df_X,pd.DataFrame({"lineage": ["unidentified_residue"],"Variant_proportion": [residue],}),],ignore_index=True,)
    return df_X

def save_outputs(
    out_dir: str | Path,
    snv_sample_csv: str | Path,
    df_sample_snv: pd.DataFrame,
    df_matched: pd.DataFrame,
    df_square: pd.DataFrame,
    matrixE_square_df: pd.DataFrame,
    E_inv: pd.DataFrame,
    R: pd.Series,
    df_X: pd.DataFrame,
) -> Dict[str, Path]:
    """
    主要出力を out_dir に保存する。
    すべてのファイル名に snv_sample の元ファイル名（stem）を含める。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(snv_sample_csv).stem
    paths: Dict[str, Path] = {}
    def _save(df: pd.DataFrame, name: str) -> None:
        p = out_dir / f"{stem}_{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
    if debug:
        _save(df_sample_snv, "df_sample_snv")
        _save(df_matched, "df_matched")
        # df_square, matrixE_square_df, E_inv, R, df_X
        _save(df_square, "df_square")
        matrixE_out = matrixE_square_df.reset_index(drop=True)
        matrixE_out.to_csv(out_dir / f"{stem}_matrixE_square.csv", index=False)
        paths["matrixE_square"] = out_dir / f"{stem}_matrixE_square.csv"
        # E_inv は系統×行 の形（見やすいように行/列名を保存）
        E_inv.to_csv(out_dir / f"{stem}_E_inv.csv")
        paths["E_inv"] = out_dir / f"{stem}_E_inv.csv"
        # R（AF）保存
        pd.DataFrame({"AF": R.values}).to_csv(out_dir / f"{stem}_R.csv", index=False)
        paths["R"] = out_dir / f"{stem}_R.csv"
        # 最終系統割合 X
        df_X.to_csv(out_dir / f"{stem}_variant_proportions.csv", index=False)
        paths["variant_proportions"] = out_dir / f"{stem}_variant_proportions.csv"
    return paths

# =============================================================================
# 収束ループ（要件に合わせた流れ）
#   - summarize が少なくとも1回走る
#   - 収束過程で
#       同一列統合 → 同一パターン判定 → summarize →（戻って同一列統合…）
#   - matrixE を正方化
#   - 逆行列が無ければ再度収束ループへ戻る
# =============================================================================
def run_pipeline(
    snv_sample_csv: str | Path,
    constellations_dir: str | Path,
    out_dir: str | Path,
    lineages_txt: str | Path | None = None,
    max_outer_loops: int = MAX_LOOPS_DEFAULT,
    batch_err: list[tuple[str, str, str]] | None = None,  # (sample, kind, detail)
) -> pd.DataFrame:   # ← 戻り値が df_X なので pd.DataFrame にしておくのが正しい
    snv_sample_csv = Path(snv_sample_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = snv_sample_csv.stem
    # Step 1
    df_sample_snv = concat_mutation_from_snv_sample(snv_sample_csv)
    print(f"[STEP1] concat_mutation_from_snv_sample done.")
    # ---- 途中出力 (Step 1)
    if debug:
        p_step1 = out_dir / f"{stem}_snv.csv"
        df_sample_snv.to_csv(p_step1, index=False)
        print(f"[STEP1] df_sample_snv saved: {p_step1}, shape={df_sample_snv.shape}")
        print(df_sample_snv.head())
    # Step 2
    const_sites = load_constellations_sites(constellations_dir, lineages_txt=lineages_txt)
    # --- 各系統のSNVパターン を CSV 出力（確認・記録用） ---
    if lineages_txt is not None:
        df_const_wide = constellations_to_wide_df(const_sites)
        p_const = out_dir / "lineage_snv_pattern.csv"
        df_const_wide.to_csv(p_const, index=False)
        print(f"[INFO] constellations matrix saved: {p_const}")
    df_matched = match_mutations_to_constellations(df_sample_snv, const_sites)
    # ---- 途中出力 (Step 2)
    sample_snv_dir = out_dir / "sample_snv_matched"
    sample_snv_dir.mkdir(parents=True, exist_ok=True)
    p_step2 = sample_snv_dir / f"{stem}_snv_matched.csv"
    df_matched.to_csv(p_step2, index=False)
    print(f"[STEP2] match_mutations_to_constellations done. {p_step2} saved.")
    if debug:
        print(f"[STEP2] df_matched saved: {p_step2}, shape={df_matched.shape}")
        print(df_matched.head())
    # --- (Step 3) Step2の後：同一列があれば統合、なければ skip
    lineage_cols = get_lineage_columns(df_matched)
    if has_identical_lineage_columns(df_matched, lineage_cols):
        df_after_consolidation, _ = consolidate_identical_lineage_columns(df_matched, lineage_cols)
        print("[STEP3] consolidate_identical_lineage_columns (after Step2) done.")
    else:
        df_after_consolidation = df_matched.copy()
        print("[STEP3] no identical lineage columns -> skip")
    if debug:
        print(f"        shape={df_after_consolidation.shape}")
        print(df_after_consolidation.head())
    # ここから「逆行列が作れるまで」繰り返す
    # 要件: summarize が少なくとも1回走る
    #summarized_once = False
    df_current = df_after_consolidation.copy()
    for outer in range(1, max_outer_loops + 1):
        print(f"[INFO] Outer loop: {outer}")
        # =========================================================
        # Inner（Step3/Step4）: 同一列⇄同一パターンを収束させる（ここは1箇所）
        #   仕様: summarize は outer ごとに必ず最低1回実行する
        # =========================================================
        df_summary = summarize_patterns_with_depth(df_current)
        print("[STEP4] summarize_patterns_with_depth done.")
        if debug:
            print(f"        shape={df_summary.shape}")
            print(df_summary.head())
        for _inner in range(1, 1000):
            changed = False
            # --- Step3: 同一列（系統）統合
            lineage_cols = get_lineage_columns(df_summary)
            if has_identical_lineage_columns(df_summary, lineage_cols):
                df_summary, _ = consolidate_identical_lineage_columns(df_summary, lineage_cols)
                changed = True
                print("[STEP3] consolidate_identical_lineage_columns (inner) done.")
                if debug:
                    print(f"        shape={df_summary.shape}")
                    print(df_summary.head())
            # --- Step4: 同一パターン（行）統合
            lineage_cols = get_lineage_columns(df_summary)
            if has_identical_patterns(df_summary, lineage_cols):
                df_summary = summarize_patterns_with_depth(df_summary)
                changed = True
                print("[STEP4] summarize_patterns_with_depth (inner) done.")
                if debug:
                    print(f"        shape={df_summary.shape}")
                    print(df_summary.head())
            if not changed:
                break
        # =========================================================
        # Step5: matrixE を正方化（DP が小さい行から削除）
        # =========================================================
        try:
            sq = build_matrixE_and_square(df_summary, drop_by="DP")
            print("[STEP5] build_matrixE_and_square done.")
            if debug:
                print(f"        shape={sq.matrixE_square_df.shape}")
                print(sq.df_square.head())
                p_step5 = out_dir / f"debug_{stem}_matrixE.csv"
                sq.df_square.to_csv(p_step5, index=False)
        except ValueError as e:
            print(f"[WARN] Square build failed: {e}")
            raise
        # =========================================================
        # Step5 後の整形（重要）:
        #   Step5 の行削除で「全て0の系統列」が新規に生まれ得るため
        #   Step5 -> 全0列削除 -> Step3（同一列統合）
        #   ここで列構造が変わったら outer 先頭（inner入口）に戻す
        # =========================================================
        df_next = sq.df_square.copy()
        # --- 全て0の系統列を削除（標準出力には出さず、batch_err に記録）
        lineage_cols_next = get_lineage_columns(df_next)
        zero_lineages = [c for c in lineage_cols_next if (df_next[c] == 0).all()]
        if zero_lineages:
            df_next = df_next.drop(columns=zero_lineages)
            if batch_err is not None:
                batch_err.append((stem, "removed_zero_lineages", ",".join(zero_lineages)))
        # --- Step3: 同一列（系統）統合（Step5後）
        lineage_cols_next = get_lineage_columns(df_next)
        if has_identical_lineage_columns(df_next, lineage_cols_next):
            df_next, _ = consolidate_identical_lineage_columns(df_next, lineage_cols_next)
            print("[STEP3] consolidate_identical_lineage_columns (after Step5) done.")
            # 列構造が変わった可能性が高いので outer 先頭へ戻す
            df_current = df_next
            continue
        # 全0列削除が発生していれば列構造が変わっているので outer 先頭へ戻す
        if zero_lineages:
            df_current = df_next
            continue
        # =========================================================
        # 逆行列判定
        # =========================================================
        E_inv = try_inverse(sq.matrixE_square_df)
        if E_inv is None:
            print("[INFO] Inverse not available -> continue outer loop with updated df.")
            # 重要：df_summary に戻さず、Step5後の df_next を持ち越して進捗を残す
            df_current = df_next
            continue
        # 逆行列あり：E_inv と R を出力し、最終Xも計算して出力
        df_X = estimate_variant_proportions(E_inv, sq.R)
        if debug:
            paths = save_outputs(
                out_dir=out_dir,
                snv_sample_csv=snv_sample_csv,
                df_sample_snv=df_sample_snv,
                df_matched=df_matched,
                df_square=sq.df_square,
                matrixE_square_df=sq.matrixE_square_df,
                E_inv=E_inv,
                R=sq.R,
                df_X=df_X,
            )
            print("[DONE] Debug outputs saved:")
            for k, p in paths.items():
                print(f"  - {k}: {p}")
        else:
            print("[DONE] Finished")
        return df_X
    raise RuntimeError(
        f"Failed to obtain invertible square matrix within max_outer_loops={max_outer_loops}."
    )

# =============================================================================
# バッチ処理関数（フォルダ内 *.csv を全部処理→一覧表CSV）
# =============================================================================
def run_batch(
    input_dir: str | Path,
    constellations_dir: str | Path,
    out_dir: str | Path,
    lineages_txt: str | Path | None = None,
    max_outer_loops: int = MAX_LOOPS_DEFAULT,
    summary_csv_name: str = "variant_proportions_matrix.csv",
) -> Path:
    # -------------------------------------------------------------------------
    #    input_dir 内の *.csv を順次 sample_snv として処理し、
    #    最終的に「行=系統、列=サンプル名」の一覧表CSVを out_dir に保存する。
    # -------------------------------------------------------------------------
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"input_dir must be an existing directory: {input_dir}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_files = sorted(list(input_dir.glob("*.csv")) + list(input_dir.glob("*.vcf")))
    if not input_files:
        raise ValueError(f"No .csv or .vcf files found in: {input_dir}")
    series_list: list[pd.Series] = []
    batch_err: list[tuple[str, str, str]] = []
    for fp in input_files:
        sample = fp.stem
        print("=" * 80)
        print(f"[BATCH] Processing: {fp}  (sample={sample})")
        try:
            df_X = run_pipeline(
                snv_sample_csv=fp,
                constellations_dir=constellations_dir,
                out_dir=out_dir,
                lineages_txt=lineages_txt,
                max_outer_loops=max_outer_loops,
                batch_err=batch_err,
            )
            # df_X: columns = ["lineage","Variant_proportion"]
            s = df_X.set_index("lineage")["Variant_proportion"]
            s.name = sample
            series_list.append(s)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            batch_err.append((sample, "error", msg))
            print(f"[BATCH][ERROR] {sample}: {msg}")
    if not series_list:
        raise RuntimeError("All samples failed; no output matrix created.")
    # 行=系統、列=サンプル名 に整形
    df_mat = pd.concat(series_list, axis=1).sort_index(axis=0)
    # NaN（出現しない系統）を 0 に
    df_mat = df_mat.fillna(0.0)
    # 出力
    out_path = out_dir / summary_csv_name
    df_mat.to_csv(out_path, index=True)
    print("=" * 80)
    print(f"[BATCH] Saved variant proportion matrix: {out_path} (#lineages, #samples)={df_mat.shape}\n")
    print(df_mat.head())
    print("-" * 80)
    # -------------------------------------------------------------------------
    #   sample_snv/ 内の *_snv_lineage.csv を集約し、Lineageごとの SNV×Sample 行列を作成
    #   - 行: Mutation (SNV)
    #   - 列: sample 名
    #   - 値: DP及びAF（allele frequency = DPalt/DP）
    #   出力: out_dir/lineage_matched/
    # -------------------------------------------------------------------------
    sample_snv_dir = out_dir / "sample_snv_matched"
    lineage_matched_dir = out_dir / "lineage_snv_matched"
    lineage_matched_dir.mkdir(parents=True, exist_ok=True)
    step2_files = sorted(sample_snv_dir.glob("*_snv_matched.csv"))
    if step2_files:
        # {lineage: {sample: Series(index=Mutation, value=AF)}}
        per_lineage_af: dict[str, dict[str, pd.Series]] = {}
        per_lineage_dp: dict[str, dict[str, pd.Series]] = {}
        for fp in step2_files:
            sample = fp.name.replace("_snv_matched.csv", "")
            df2 = pd.read_csv(fp)
            # 必須列(Mutation)チェック
            required = {"Mutation", "DP", "DPalt"}
            if not required.issubset(df2.columns):
                print(f"[WARN] {fp.name}: missing columns {sorted(required - set(df2.columns))}. Skipped.")
                continue
            # AF列の準備。DPalt/DP から計算
            dp = pd.to_numeric(df2["DP"], errors="coerce")
            dpalt = pd.to_numeric(df2["DPalt"], errors="coerce")
            df2["AF"] = np.where((dp > 0) & dp.notna() & dpalt.notna(), dpalt / dp, 0.0)
            # DP列（depth）を数値化
            df2["DP"] = pd.to_numeric(df2["DP"], errors="coerce")
            # lineage列を取得
            lineage_cols = get_lineage_columns(df2)
            if not lineage_cols:
                print(f"[WARN] {fp.name}: No lineage columns found. Skipped.")
                continue
            # lineageごとに、該当行（lineage==1）の Mutation -> AFとDPを格納
            for lin in lineage_cols:
                # lineage 列が 1 （変異が存在する）の行だけを抽出
                df_l = df2.loc[df2[lin] == 1, ["Mutation", "AF", "DP"]].copy()
                if df_l.empty:
                    continue
                # --- AF: Mutation → AF の Series を作る
                s_af = df_l.groupby("Mutation")["AF"].max()
                s_af.name = sample
                per_lineage_af.setdefault(lin, {})[sample] = s_af
                # --- DP: Mutation → DP の Series を作る（あれば）
                s_dp = df_l.groupby("Mutation")["DP"].max()
                s_dp.name = sample
                per_lineage_dp.setdefault(lin, {})[sample] = s_dp
        # 出力
        for lin, sample_map_af in per_lineage_af.items():
            if not sample_map_af:
                continue
            # --- AF
            df_af = pd.concat(sample_map_af.values(), axis=1)
            df_af = df_af.reindex(_sort_mutations(list(df_af.index)))
            df_af = df_af.fillna(0.0).astype(float)
            out_af = lineage_matched_dir / f"{lin}_snv_matched_af.csv"
            df_af.to_csv(out_af, index=True)
            # --- DP
            sample_map_dp = per_lineage_dp[lin]  # ← ここで KeyError が出るなら upstream の構築が不整合
            df_dp = pd.concat(sample_map_dp.values(), axis=1)
            # AF と「行」「列」を完全に揃える（これが一番事故りにくい）
            df_dp = df_dp.reindex(index=df_af.index, columns=df_af.columns)
            df_dp = df_dp.fillna(0).round().astype(int)
            out_dp = lineage_matched_dir / f"{lin}_snv_matched_dp.csv"
            df_dp.to_csv(out_dp, index=True)
        print(f"[BATCH] lineage SNV matrix saved in {lineage_matched_dir}")

    # -------------------------------------------------------------------------
    #   batch_err（error + removed_zero_lineages）をまとめて出力
    # -------------------------------------------------------------------------
    if batch_err:
        err_path = out_dir / "batch_errors.tsv"
        df_err=pd.DataFrame(batch_err, columns=["sample", "kind", "detail"])
        df_err.to_csv(err_path, sep="\t", index=False)
        print(f"[BATCH] batch_err saved: {err_path}")
        print(df_err)
    return out_path

# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """
    Usage:
      # 1ファイル
      python IMR.py sample_snv.csv constellations -l lineages.txt -o out
      # フォルダ一括
      python IMR.py dir_samples/ constellations -l lineages.txt -o out
    """
    parser = argparse.ArgumentParser(
        description="Variant proportion estimation pipeline (single file or batch folder)."
    )
    parser.add_argument("snv_input", help="snv_sample CSV/VCF file OR directory containing CSV/VCF files")
    parser.add_argument("constellations", help="constellations directory containing lineage JSONs")
    parser.add_argument("-o", "--out-dir", dest="out_dir", default=".", help="output directory")
    parser.add_argument("-l", "--lineages", dest="lineages", default=None,
                        help="optional lineages.txt (limit target lineages)")
    parser.add_argument("-m", "--max-loops", dest="max_loops", type=int, default=MAX_LOOPS_DEFAULT,
                        help="max outer loops to search invertible matrix")
    args = parser.parse_args()
    snv_input = Path(args.snv_input)
    if snv_input.is_dir():
        run_batch(
            input_dir=snv_input,
            constellations_dir=args.constellations,
            out_dir=args.out_dir,
            lineages_txt=args.lineages,
            max_outer_loops=args.max_loops,
            summary_csv_name="variant_proportions_matrix.csv",
        )
    else:
        # 1ファイル処理（従来と同じ）
        run_pipeline(
            snv_sample_csv=snv_input,
            constellations_dir=args.constellations,
            out_dir=args.out_dir,
            lineages_txt=args.lineages,
            max_outer_loops=args.max_loops,
        )

if __name__ == "__main__":
    main()
