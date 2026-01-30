#!/usr/bin/env python3
"""
Schema Preprocessor for Graph Construction and Similarity Checking.

Transforms parsed schema into two lists per table:
1. Semantic Names (Embedding Target): FQN with PK/FK annotations and example data
2. Structural Types (Constraint Vector): Data types for each column

This preprocessing is optimized for:
- Vector similarity search
- Reinforced Edge Strategy (FK + semantic similarity)
- IND (Inclusion Dependency) inference via example data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from src.common.types import SchemaInfo, TableNode


@dataclass
class PreprocessedColumn:
    """Represents a preprocessed column with all metadata."""
    table_name: str
    column_name: str
    data_type: str
    is_primary_key: bool = False
    foreign_key_ref: Optional[str] = None  # Format: "target_table.target_column"
    fk_points_to_pk: bool = True  # True if FK references a PK column (higher weight in graph)
    examples: List[Any] = field(default_factory=list)
    
    def to_semantic_name(
        self,
        include_examples: bool = False,
        fk_in_semantics: bool = False,
    ) -> str:
        """
        Generate the semantic name string for embedding.
        
        Args:
            include_examples: If True, append (e.g. 'val1', 'val2') from column examples.
            fk_in_semantics: If True, append (references target.column) for FK columns.
        """
        parts = [f"{self.table_name}.{self.column_name}"]
        
        tags = []
        if self.is_primary_key:
            tags.append("PK")
        if fk_in_semantics and self.foreign_key_ref:
            tags.append(f"references {self.foreign_key_ref}")
        
        if tags:
            parts.append(f"({', '.join(tags)})")
        
        if include_examples and self.examples:
            formatted_examples = []
            for ex in self.examples[:3]:
                if isinstance(ex, str):
                    formatted_examples.append(f"'{ex}'")
                else:
                    formatted_examples.append(str(ex))
            parts.append(f"(e.g. {', '.join(formatted_examples)})")
        
        return " ".join(parts)


@dataclass
class PreprocessedTable:
    """Represents a preprocessed table with semantic names and structural types."""
    table_name: str
    context: str  # Brief description of table's domain/purpose
    semantic_names: List[str] = field(default_factory=list)
    structural_types: List[str] = field(default_factory=list)
    columns: List[PreprocessedColumn] = field(default_factory=list)


@dataclass 
class PreprocessedSchema:
    """Represents a fully preprocessed database schema."""
    db_id: str
    tables: Dict[str, PreprocessedTable]
    # FK mapping for quick lookup: {(table, column): (ref_table, ref_column)}
    fk_mapping: Dict[Tuple[str, str], Tuple[str, str]] = field(default_factory=dict)


def _infer_table_context(table_node: TableNode) -> str:
    """
    Infer the table's domain/context from its name and columns.
    
    Args:
        table_node: Parsed table information
        
    Returns:
        Brief context description string
    """
    name_lower = table_node.name.lower()
    
    # Common domain patterns
    domain_hints = {
        "customer": "Customer/Client management",
        "order": "Order/Transaction processing",
        "product": "Product/Inventory management",
        "transaction": "Financial/Transaction tracking",
        "account": "Account/Financial management",
        "user": "User management",
        "employee": "HR/Employee management",
        "invoice": "Billing/Invoice management",
        "payment": "Payment processing",
        "stock": "Inventory/Stock management",
        "sales": "Sales tracking",
        "yearmonth": "Financial/Consumption tracking",
        "district": "Geographic/Location data",
        "card": "Card/Payment method",
        "loan": "Loan/Credit management",
        "client": "Client management",
    }
    
    for hint_key, context in domain_hints.items():
        if hint_key in name_lower:
            return context
    
    # Fallback: use column names to infer
    col_str = " ".join(table_node.columns).lower()
    
    if "price" in col_str or "amount" in col_str or "cost" in col_str:
        return "Financial data"
    if "date" in col_str or "time" in col_str:
        return "Time-series/Historical data"
    if "name" in col_str or "address" in col_str:
        return "Entity/Profile data"
    
    return "General data"


def preprocess_schema(
    schema_info: SchemaInfo,
    include_examples: bool = False,
    fk_in_semantics: bool = False,
) -> PreprocessedSchema:
    """
    Preprocess a parsed schema into the format optimized for graph construction.
    
    Args:
        schema_info: Parsed schema information
        include_examples: If True, add (e.g. ...) to semantic names. Default False.
        fk_in_semantics: If True, add (references x.y) to semantic names. Default False.
        
    Returns:
        PreprocessedSchema with semantic names and structural types per table
    """
    preprocessed_tables = {}
    global_fk_mapping = {}
    
    for table_name, table_node in schema_info.tables.items():
        pk_set = {pk.upper() for pk in table_node.primary_keys}
        
        # Use parser-provided FK details and column types (no DDL re-parsing)
        fk_column_refs = table_node.fk_column_refs
        col_types_list = table_node.column_types
        
        preprocessed_columns = []
        semantic_names = []
        structural_types = []
        
        for i, col_name in enumerate(table_node.columns):
            is_pk = col_name.upper() in pk_set
            
            # FK reference from parser
            fk_ref = None
            fk_points_to_pk = True
            col_upper = col_name.upper()
            if col_upper in fk_column_refs:
                ref_table, ref_col = fk_column_refs[col_upper]
                fk_ref = f"{ref_table}.{ref_col}"
                global_fk_mapping[(table_name, col_name)] = (ref_table, ref_col)
                # Check if target column is PK (for future graph weight boost)
                ref_table_node = schema_info.tables.get(ref_table)
                if ref_table_node:
                    ref_pk_set = {pk.upper() for pk in ref_table_node.primary_keys}
                    fk_points_to_pk = ref_col.upper() in ref_pk_set
                else:
                    fk_points_to_pk = False
            
            examples = table_node.column_examples[i] if i < len(table_node.column_examples) else []
            data_type = col_types_list[i] if i < len(col_types_list) else "unknown"
            
            prep_col = PreprocessedColumn(
                table_name=table_name,
                column_name=col_name,
                data_type=data_type,
                is_primary_key=is_pk,
                foreign_key_ref=fk_ref,
                fk_points_to_pk=fk_points_to_pk,
                examples=examples,
            )
            
            preprocessed_columns.append(prep_col)
            semantic_names.append(
                prep_col.to_semantic_name(
                    include_examples=include_examples,
                    fk_in_semantics=fk_in_semantics,
                )
            )
            structural_types.append(data_type)
        
        # Infer table context
        context = _infer_table_context(table_node)
        
        preprocessed_tables[table_name] = PreprocessedTable(
            table_name=table_name,
            context=context,
            semantic_names=semantic_names,
            structural_types=structural_types,
            columns=preprocessed_columns,
        )
    
    return PreprocessedSchema(
        db_id=schema_info.db_id,
        tables=preprocessed_tables,
        fk_mapping=global_fk_mapping,
    )


def serialize_preprocessed_table(prep_table: PreprocessedTable) -> str:
    """
    Serialize a preprocessed table for embedding.
    
    Combines context and semantic names into a single embedding-ready string.
    
    Args:
        prep_table: Preprocessed table
        
    Returns:
        String optimized for semantic embedding
    """
    lines = [
        f"Table Context: {prep_table.table_name} ({prep_table.context})",
        "Columns:"
    ]
    
    for sem_name in prep_table.semantic_names:
        lines.append(f"  - {sem_name}")
    
    return "\n".join(lines)


def serialize_preprocessed_schema(prep_schema: PreprocessedSchema) -> Dict[str, str]:
    """
    Serialize all tables in a preprocessed schema.
    
    Args:
        prep_schema: Preprocessed schema
        
    Returns:
        Dict mapping table name to serialized text
    """
    return {
        table_name: serialize_preprocessed_table(prep_table)
        for table_name, prep_table in prep_schema.tables.items()
    }


def get_type_compatible_pairs(prep_schema: PreprocessedSchema) -> List[Tuple[str, str, str, str]]:
    """
    Get all column pairs that have compatible types (potential soft edges).
    
    This is used as a hard constraint filter before computing semantic similarity.
    Only columns with the same structural type can form edges.
    
    Args:
        prep_schema: Preprocessed schema
        
    Returns:
        List of (table1.col1, type1, table2.col2, type2) tuples where types match
    """
    compatible_pairs = []
    
    # Collect all columns with their types
    all_columns = []
    for table_name, prep_table in prep_schema.tables.items():
        for col in prep_table.columns:
            all_columns.append((
                f"{table_name}.{col.column_name}",
                col.data_type,
                table_name,
            ))
    
    # Find compatible pairs (same type, different tables)
    for i in range(len(all_columns)):
        for j in range(i + 1, len(all_columns)):
            col1, type1, table1 = all_columns[i]
            col2, type2, table2 = all_columns[j]
            
            # Only consider cross-table pairs with same type
            if table1 != table2 and type1 == type2:
                compatible_pairs.append((col1, type1, col2, type2))
    
    return compatible_pairs


# For convenient access to preprocessed data
def to_dict(prep_schema: PreprocessedSchema) -> Dict[str, Any]:
    """
    Convert PreprocessedSchema to a dictionary format.
    
    Useful for JSON serialization or debugging.
    
    Args:
        prep_schema: Preprocessed schema
        
    Returns:
        Dictionary representation
    """
    result = {
        "db_id": prep_schema.db_id,
        "tables": {},
        "fk_mapping": {
            f"{t}.{c}": f"{rt}.{rc}" 
            for (t, c), (rt, rc) in prep_schema.fk_mapping.items()
        },
    }
    
    for table_name, prep_table in prep_schema.tables.items():
        result["tables"][table_name] = {
            "context": prep_table.context,
            "semantic_names": prep_table.semantic_names,
            "structural_types": prep_table.structural_types,
        }
    
    return result
