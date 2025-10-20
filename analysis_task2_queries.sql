-- Auto-generated queries for `arrangements_clean` (MariaDB/XAMPP friendly)

-- (a)
SELECT place AS mesto, COUNT(*) AS broj_aranzmana FROM arrangements_clean GROUP BY place ORDER BY broj_aranzmana DESC, mesto ASC;

-- (b)
SELECT COUNT(*) AS br_4i5 FROM arrangements_clean WHERE stars IN (4,5);

-- (c)
SELECT COUNT(*) AS br_all_inclusive FROM arrangements_clean WHERE (is_all_inclusive = 1) OR (board REGEXP 'all[[:space:]]*inclusive');

-- (d)
SELECT COUNT(*) AS br_ge_12_nocenja FROM arrangements_clean WHERE nights >= 12;

-- (e)
SELECT * FROM arrangements_clean ORDER BY price_eur_norm DESC LIMIT 30;

-- (f)
SELECT * FROM arrangements_clean WHERE (is_all_inclusive = 1) OR (board REGEXP 'all[[:space:]]*inclusive') ORDER BY price_eur_norm DESC LIMIT 30;
