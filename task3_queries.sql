-- Task 3 queries for `arrangements_clean` (MariaDB/XAMPP friendly)

    -- (a) 10 najzastupljenijih mesta (po broju aranžmana)
    SELECT place AS mesto, COUNT(*) AS broj
FROM arrangements_clean
GROUP BY place
ORDER BY broj DESC, mesto ASC;

    -- klijent prikazuje TOP 10 iz rezultata

    -- (b) broj aranžmana po mestima (full tabela, često velika)
    SELECT place AS mesto, COUNT(*) AS broj
FROM arrangements_clean
GROUP BY place
ORDER BY broj DESC, mesto ASC;


    -- (c) broj i procenat po zvezdicama
    SELECT
  stars,
  COUNT(*) AS broj,
  100.0 * COUNT(*) / (SELECT COUNT(*) FROM arrangements_clean) AS procenat
FROM arrangements_clean
GROUP BY stars
ORDER BY stars;


    -- (d) broj i procenat po cenovnim opsezima (EUR total, price_eur_norm)
    SELECT bucket, COUNT(*) AS broj,
       100.0 * COUNT(*) / (SELECT COUNT(*) FROM arrangements_clean) AS procenat
FROM (
  SELECT CASE
           WHEN price_eur_norm IS NULL THEN 'Unknown'
           WHEN price_eur_norm <= 500 THEN '<= 500'
           WHEN price_eur_norm BETWEEN 501 AND 1500 THEN '501-1500'
           WHEN price_eur_norm BETWEEN 1501 AND 3000 THEN '1501-3000'
           WHEN price_eur_norm >= 3000 THEN '>= 3000'
           ELSE 'Other'
         END AS bucket
  FROM arrangements_clean
) t
GROUP BY bucket
ORDER BY
  CASE bucket
    WHEN '<= 500' THEN 1
    WHEN '501-1500' THEN 2
    WHEN '1501-3000' THEN 3
    WHEN '>= 3000' THEN 4
    WHEN 'Unknown' THEN 5
    ELSE 6
  END;


    -- (e) broj i procenat po tipu usluge (normalizovan board -> service_type)
    SELECT service_type, COUNT(*) AS broj,
       100.0 * COUNT(*) / (SELECT COUNT(*) FROM arrangements_clean) AS procenat
FROM (
  SELECT
    CASE
      WHEN board IS NULL OR TRIM(board) = '' THEN 'Other/Unknown'
      WHEN LOWER(board) REGEXP 'ultra[[:space:]]*all[[:space:]]*inclusive' THEN 'Ultra All Inclusive'
      WHEN LOWER(board) REGEXP 'all[[:space:]]*inclusive' THEN 'All Inclusive'
      WHEN LOWER(board) REGEXP 'full[[:space:]]*board|\bfb\b' THEN 'Full Board'
      WHEN LOWER(board) REGEXP 'half[[:space:]]*board|\bhb\b|polu' THEN 'Half Board'
      WHEN LOWER(board) REGEXP 'bed[[:space:]]*&[[:space:]]*breakfast|breakfast|\bbb\b' THEN 'Bed & Breakfast'
      WHEN LOWER(board) REGEXP 'room[[:space:]]*only|\bro\b|self[[:space:]]*cater' THEN 'Room Only / Self-catering'
      ELSE 'Other/Unknown'
    END AS service_type
  FROM arrangements_clean
) x
GROUP BY service_type
ORDER BY broj DESC, service_type ASC;
