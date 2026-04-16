"""
query_examples.py — Useful queries against liberation_day.duckdb
Run:  python database/query_examples.py
"""
import os
import duckdb

DB_PATH = os.path.join(os.path.dirname(__file__), 'liberation_day.duckdb')
con = duckdb.connect(DB_PATH, read_only=True)

def q(title, sql):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")
    print(con.execute(sql).fetchdf().to_string(index=False))

# 1. Top 10 countries by US tariff rate
q("Top 10 countries facing highest US Liberation Day tariffs","""
    SELECT country_name, iso3, tariff_pct
    FROM tariffs ORDER BY tariff_pct DESC LIMIT 10
""")

# 2. US welfare across all 9 scenarios
q("US welfare change (%) across all 9 tariff scenarios","""
    SELECT scenario_id, scenario_name,
           ROUND(welfare_pct,3)       AS welfare_pct,
           ROUND(cpi_pct,3)           AS cpi_pct,
           ROUND(employment_pct,3)    AS employment_pct,
           ROUND(trade_deficit_pct,3) AS deficit_pct
    FROM ge_results
    WHERE iso3 = 'USA'
    ORDER BY scenario_id
""")

# 3. Who wins and who loses? (Liberation Day, no retaliation)
q("Top 10 winners & losers — Liberation Day scenario","""
    (SELECT country_name, iso3, ROUND(welfare_pct,3) AS welfare_pct, 'Winner' AS status
     FROM ge_results WHERE scenario_id = 0 AND iso3 != 'USA'
     ORDER BY welfare_pct DESC LIMIT 10)
    UNION ALL
    (SELECT country_name, iso3, ROUND(welfare_pct,3), 'Loser'
     FROM ge_results WHERE scenario_id = 0 AND iso3 != 'USA'
     ORDER BY welfare_pct ASC LIMIT 10)
    ORDER BY welfare_pct DESC
""")

# 4. Top US import partners by trade value (BACI 2023)
q("Top 15 US import partners by trade value (BACI 2023)","""
    SELECT importer_iso3, exporter_iso3,
           ROUND(SUM(value_1000usd)/1e6, 1) AS trade_billion_usd
    FROM baci_trade
    WHERE importer_iso3 = 'USA'
    GROUP BY importer_iso3, exporter_iso3
    ORDER BY trade_billion_usd DESC
    LIMIT 15
""")

# 5. Top traded HS6 products into the US
q("Top 10 HS6 product categories exported to US (BACI 2023)","""
    SELECT hs6_product_code,
           ROUND(SUM(value_1000usd)/1e6, 1) AS trade_billion_usd,
           COUNT(DISTINCT exporter_iso3)    AS n_exporters
    FROM baci_trade
    WHERE importer_iso3 = 'USA'
    GROUP BY hs6_product_code
    ORDER BY trade_billion_usd DESC
    LIMIT 10
""")

# 6. US–China bilateral trade by broad sector (ITPDS)
q("US–China bilateral trade by sector (ITPDS, latest year)","""
    SELECT broad_sector, year,
           ROUND(SUM(trade_usd_millions),1) AS trade_usd_millions
    FROM itpds_trade
    WHERE ((exporter_iso3='USA' AND importer_iso3='CHN')
        OR (exporter_iso3='CHN' AND importer_iso3='USA'))
      AND year = (SELECT MAX(year) FROM itpds_trade)
    GROUP BY broad_sector, year
    ORDER BY trade_usd_millions DESC
""")

# 7. Surrogate model: US welfare by tariff rate (no retaliation)
q("Surrogate: US welfare vs tariff rate (no partner retaliation)","""
    SELECT us_tariff,
           ROUND(AVG(welfare_us),3)       AS welfare_us,
           ROUND(AVG(cpi_us),2)           AS cpi_us,
           ROUND(AVG(employment_us),3)    AS employment_us
    FROM surrogate_training
    WHERE china_rate = 0 AND eu_rate = 0 AND canmex_rate = 0
    GROUP BY us_tariff
    ORDER BY us_tariff
""")

# 8. Gravity: distance and trade relationship (US partners)
q("Gravity variables for top US trade partners","""
    SELECT g.iso3_d AS partner, ROUND(g.distance,0) AS distance_km,
           CAST(g.member_wto_d AS INTEGER) AS wto_member,
           CAST(g.member_eu_d  AS INTEGER) AS eu_member,
           ROUND(b.trade_billion_usd,1) AS imports_from_usd_bn
    FROM gravity g
    JOIN (
        SELECT exporter_iso3,
               ROUND(SUM(value_1000usd)/1e6,1) AS trade_billion_usd
        FROM baci_trade WHERE importer_iso3='USA'
        GROUP BY exporter_iso3
    ) b ON g.iso3_d = b.exporter_iso3
    WHERE g.iso3_o = 'USA' AND g.year = 2019
    ORDER BY b.trade_billion_usd DESC
    LIMIT 15
""")

# 9. OECD IO: largest inter-country supply chain flows (2022)
q("Largest cross-country IO supply chain flows (OECD ICIO 2022)","""
    SELECT source_country, dest_country,
           source_sector, dest_sector,
           ROUND(SUM(flow_usd_millions),1) AS flow_usd_millions
    FROM oecd_icio
    WHERE year = 2022 AND source_country != dest_country
    GROUP BY source_country, dest_country, source_sector, dest_sector
    ORDER BY flow_usd_millions DESC
    LIMIT 10
""")

# 10. Surrogate validation summary
q("Surrogate bias correction: raw vs corrected vs true welfare_us","""
    SELECT scenario_name,
           ROUND(welfare_us_true,3)  AS true_ge,
           ROUND(welfare_us_pred,3)  AS raw_surrogate,
           ROUND(welfare_us_error,3) AS error
    FROM surrogate_validation
    ORDER BY scenario_id
""")

con.close()
print("\n[Done]")
