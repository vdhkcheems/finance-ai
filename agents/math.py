import ast
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple, List

import numpy as np
import pandas as pd


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    return name.strip().lower().replace("-", "_").replace(" ", "_")


ALLOWED_FUNC_MAP = {
    "abs": np.abs,
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "min": np.minimum,
    "max": np.maximum,
}


class _SafeEvaluator(ast.NodeVisitor):
    """
    Safely evaluate a math expression AST over pandas Series variables.
    Supports +, -, *, /, **, unary +/-, parentheses, and a small set of functions.
    """

    def __init__(self, variables: Dict[str, pd.Series]):
        self.variables = variables

    def visit(self, node):  # type: ignore[override]
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Num):  # py<3.8
            return node.n
        if isinstance(node, ast.Constant):  # numbers only
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed in formulas")
        if isinstance(node, ast.Name):
            name = _normalize_name(node.id)
            if name not in self.variables:
                raise KeyError(f"Unknown variable '{name}' in formula")
            return self.variables[name]
        if isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError("Unsupported binary operator in formula")
        if isinstance(node, ast.UnaryOp):
            operand = self.visit(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator in formula")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed")
            func_name = _normalize_name(node.func.id)
            if func_name not in ALLOWED_FUNC_MAP:
                raise ValueError(f"Function '{func_name}' is not allowed")
            func = ALLOWED_FUNC_MAP[func_name]
            if len(node.args) != 1:
                raise ValueError(f"Function '{func_name}' expects exactly 1 argument")
            arg_val = self.visit(node.args[0])
            return func(arg_val)
        raise ValueError("Unsupported expression in formula")


def _extract_variable_names(expr: str) -> Set[str]:
    names: Set[str] = set()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid formula syntax: {e}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(_normalize_name(node.id))
    return names


@dataclass
class Formula:
    target: str
    expression: str
    description: Optional[str] = None
    aliases: Optional[List[str]] = None


class FormulaDatabase:
    """Load and look up formulas for derived fundamentals from a text file.

    Supported file formats:
    - .json: mapping of target -> expression or objects {"expression": str, "description": str, "aliases": [..]}
    - .txt: each non-empty, non-comment line as: target = expression  # optional description
            aliases can be specified via lines like: alias: target
    - .csv: columns target, expression, description (optional), aliases (comma-separated)
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._formulas: Dict[str, Formula] = {}
        self._alias_to_target: Dict[str, str] = {}
        self._load()

    def _register(self, formula: Formula) -> None:
        key = _normalize_name(formula.target)
        self._formulas[key] = formula
        alias_list = formula.aliases or []
        for alias in alias_list:
            self._alias_to_target[_normalize_name(alias)] = key

    def _load(self) -> None:
        path_lower = self.file_path.lower()
        try:
            if path_lower.endswith(".json"):
                import json
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for target, val in data.items():
                    if isinstance(val, dict):
                        self._register(Formula(target=target, expression=val.get("expression", ""),
                                               description=val.get("description"),
                                               aliases=val.get("aliases")))
                    else:
                        self._register(Formula(target=target, expression=str(val)))
            elif path_lower.endswith(".csv"):
                df = pd.read_csv(self.file_path)
                for _, row in df.iterrows():
                    aliases = None
                    if "aliases" in df.columns and isinstance(row.get("aliases"), str):
                        aliases = [a.strip() for a in str(row["aliases"]).split(",") if a.strip()]
                    self._register(Formula(target=row["target"], expression=row["expression"],
                                           description=row.get("description"), aliases=aliases))
            else:
                # .txt format: target = expression  # optional description
                # alias: target  (allows mapping alias -> target)
                with open(self.file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        raw = line.strip()
                        if not raw or raw.startswith("#"):
                            continue
                        if ":" in raw and "=" not in raw:
                            # alias mapping
                            alias_name, target = [p.strip() for p in raw.split(":", 1)]
                            self._alias_to_target[_normalize_name(alias_name)] = _normalize_name(target)
                            continue
                        # split off comment
                        if "#" in raw:
                            raw, desc = raw.split("#", 1)
                            description = desc.strip()
                        else:
                            description = None
                        if "=" not in raw:
                            continue
                        target, expr = [p.strip() for p in raw.split("=", 1)]
                        self._register(Formula(target=target, expression=expr, description=description))
        except FileNotFoundError:
            # no formulas file; keep empty and allow runtime defaults
            self._formulas = {}
            self._alias_to_target = {}

    def list_targets(self) -> List[str]:
        return sorted(list(self._formulas.keys()))

    def resolve_target_key(self, name: str) -> Optional[str]:
        norm = _normalize_name(name)
        if norm in self._formulas:
            return norm
        if norm in self._alias_to_target:
            return self._alias_to_target[norm]
        # try some common synonyms
        synonyms = {
            "pat": "net_profit",
            "pat_margin": "net_profit_margin",
            "revenue": "sales",
            "profit": "net_profit",
            "de_ratio": "debt_to_equity",
        }
        if norm in synonyms:
            mapped = _normalize_name(synonyms[norm])
            if mapped in self._formulas:
                return mapped
        return None

    def get_formula(self, name: str) -> Optional[Formula]:
        key = self.resolve_target_key(name)
        if key is None:
            return None
        return self._formulas.get(key)


class MathAgent:
    """
    Agent that can compute derived fundamentals from a base DataFrame using
    formulas loaded from a lightweight on-disk database (text/JSON/CSV).
    """

    def __init__(self, formulas_path: str = "data/formulas.txt"):
        self.db = FormulaDatabase(formulas_path)

        # map common synonyms of dataset columns (left = synonym in formulas/user, right = dataset column)
        self.synonym_to_column = {
            "pat": "net_profit",
            "net_profit": "net_profit",
            "profit_after_tax": "net_profit",
            "revenue": "sales",
            "sales": "sales",
            "turnover": "sales",
            "eps": "eps",
            "roe": "roe",
            "debt_equity": "debt_equity",
            "de_ratio": "debt_equity",
            "dividend_yield": "dividend_yield",
            "avg_price": "avg_price",
            # add more base-metric synonyms here as needed
        }

    def _map_variable_to_column(self, token: str, df_columns: Set[str]) -> Optional[str]:
        token_norm = _normalize_name(token)
        # direct match
        if token_norm in df_columns:
            return token_norm
        # synonym match
        mapped = self.synonym_to_column.get(token_norm)
        if mapped and mapped in df_columns:
            return mapped
        return None

    def evaluate_on_dataframe(self, df_filtered: pd.DataFrame, expression: str) -> Tuple[pd.Series, List[str]]:
        """Evaluate expression over df rows using available columns; return series and used columns.
        Raises ValueError if required variables are missing.
        """
        expr = expression.strip()
        required_names = _extract_variable_names(expr)
        available_cols = {c for c in df_filtered.columns}
        var_map: Dict[str, pd.Series] = {}
        used_cols: List[str] = []
        for name in required_names:
            col = self._map_variable_to_column(name, available_cols)
            if not col:
                raise ValueError(f"Required variable '{name}' not found in data for expression '{expression}'")
            var_map[name] = df_filtered[col]
            used_cols.append(col)

        # Safely evaluate
        tree = ast.parse(expr, mode="eval")
        evaluator = _SafeEvaluator(var_map)
        result = evaluator.visit(tree)
        # handle divide by zero, etc.
        if isinstance(result, pd.Series):
            result = result.replace([np.inf, -np.inf], np.nan)
        return result, used_cols

    def compute_derived_metric(self, df_filtered: pd.DataFrame, metric_name: str) -> Optional[pd.DataFrame]:
        formula = self.db.get_formula(metric_name)
        if not formula:
            return None
        try:
            series, used_cols = self.evaluate_on_dataframe(df_filtered, formula.expression)
        except Exception:
            return None

        # Build long-format DataFrame like extract_fundamental_data
        out = df_filtered[["company", "sector", "period"]].copy()
        out["value"] = series
        out["metric"] = _normalize_name(metric_name)
        # Drop rows with NaN values to be consistent with extract_fundamental_data
        out = out.dropna(subset=["value"]).copy()
        # sort for consistency
        if "year" in df_filtered.columns:
            out = out.sort_values(["company", "period"]).reset_index(drop=True)
        return out

    def available_derived_metrics(self) -> List[str]:
        return self.db.list_targets()

