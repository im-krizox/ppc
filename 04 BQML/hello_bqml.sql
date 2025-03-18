CREATE OR REPLACE TABLE
  ppc_ibm.transactions AS
SELECT
  user,
  year*100+month AS month,
  use_chip,
  amount,
  merchant_state,
  CASE
    WHEN is_fraud THEN 1
    ELSE 0
END
  AS is_fraud
FROM
  ppc_ibm.txn
WHERE
  merchant_state IN ( "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY");
CREATE OR REPLACE TABLE
  ppc_ibm.tad AS
WITH
  userMonthlyAgg AS (
  SELECT
    user,
    month,
    SUM(amount) AS monto,
    COUNT(*) AS num_txn
  FROM
    ppc_ibm.transactions
  GROUP BY
    user,
    month ),
  userMonthlyAggWithWindow AS (
  SELECT
    user,
    month,
    SUM(monto) OVER (PARTITION BY user ORDER BY month ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING ) AS sum_last_6_months,
    SUM(num_txn) OVER (PARTITION BY user ORDER BY month ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING ) AS num_txn_last_6_months,
    ROW_NUMBER() OVER (PARTITION BY user ORDER BY month ) AS row_num,
    monto,
    num_txn
  FROM
    userMonthlyAgg
  ORDER BY
    user,
    month )
SELECT
  transactions.user,
  transactions.month,
  SAFE_DIVIDE(amount,SAFE_DIVIDE(sum_last_6_months,num_txn_last_6_months)) AS c_ratio_amount_vs_avg_last_6_months,
  amount AS c_amount,
  merchant_state AS d_state,
  use_chip AS d_txn_type,
  is_fraud
FROM
  userMonthlyAggWithWindow
INNER JOIN
  ppc_ibm.transactions transactions
ON
  userMonthlyAggWithWindow.user = transactions.user
  AND userMonthlyAggWithWindow.month = transactions.month
WHERE
  row_num > 6 ;
CREATE OR REPLACE MODEL
  `ppc_ibm.modelo` OPTIONS( model_type = 'LOGISTIC_REG',
    input_label_cols = ['is_fraud'],
    auto_class_weights = TRUE,
    enable_global_explain = TRUE,
    optimize_strategy='AUTO_STRATEGY',
    l1_reg=hparam_range(0,
      5),
    l2_reg=hparam_range(0,
      5),
    num_trials=5,
    DATA_SPLIT_METHOD ='RANDOM' ) AS
SELECT
  * EXCEPT(month,
    user)
FROM
  `ppc_ibm.tad`;
SELECT
  *
FROM
  ML.PREDICT(MODEL `bi-2025-01.ppc_ibm.modelo`,
    (
    SELECT
      *
    FROM
      ppc_ibm.tad
    LIMIT
      10))