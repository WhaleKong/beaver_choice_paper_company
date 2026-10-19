# Munder Difflin Multi-Agent Workflow

Four agents built with **smolagents**: one Orchestrator and three workers (Inventory, Quoting, Sales).
Every customer request comes with its request date. The Orchestrator hands the request to the workers
in a fixed order and writes the final reply to the customer.

Sources: `agent_workflow.mmd`, `request_sequence.mmd` (Mermaid). Render with
`mmdc -i <file>.mmd -o <file>.png -b white -w 1800 -s 2`.

## 1. Architecture

![Agent architecture](agent_workflow.png)

| Agent | Responsibility | Tools → starter helper |
|---|---|---|
| **Orchestrator** (`ToolCallingAgent`, `managed_agents=[inventory, quoting, sales]`) | Parse items, quantities and needed-by date; delegate Inventory → Quoting → Sales; skip Quoting and Sales when nothing can be fulfilled; write the customer reply without internal data (cash, margins) | Calls the three worker agents only |
| **Inventory Agent** | Map requested items to exact catalog names, check stock as of the request date, decide whether to restock | `find_catalog_items` → `paper_supplies`<br>`check_inventory` → `get_all_inventory`<br>`check_stock_level` → `get_stock_level`<br>`check_supplier_delivery` → `get_supplier_delivery_date`<br>`check_cash_balance` → `get_cash_balance`<br>`restock_item` → `create_transaction('stock_orders')` |
| **Quoting Agent** | Price fulfillable items, compare with similar past quotes, apply the bulk discount tier, explain the price | `search_past_quotes` → `search_quote_history`<br>`get_unit_price` → `paper_supplies`<br>`calculate_quote` → bulk discount tiers |
| **Sales Agent** | Re-verify stock, record one sale per line item, estimate delivery date, refresh the financial report | `check_stock_level` → `get_stock_level`<br>`check_delivery_date` → `get_supplier_delivery_date`<br>`fulfill_order` → `create_transaction('sales')`<br>`get_financial_report` → `generate_financial_report` |

All seven starter helpers are used by at least one tool. Data sources are the SQLite database
`munder_difflin.db` (`transactions`, `inventory`, `quotes`, `quote_requests`), the `paper_supplies`
catalog, and the supplier lead time simulated by `get_supplier_delivery_date`.

## 2. Request lifecycle

![Request sequence](request_sequence.png)

## 3. Design calls (to confirm during implementation)

- **The customer accepts the quote immediately.** The test harness has no back-and-forth, so a
  fulfillable request goes through to a recorded sale in one pass.
- **Restock rule.** Restock when stock is short of the requested quantity, or would fall below
  `min_stock_level` after the sale. Place the restock only if the supplier ETA is on or before the
  customer's needed-by date and the cash balance covers the cost. Otherwise mark the item unavailable.
- **Partial fulfilment.** Items not in the catalog (e.g. balloons, A3 paper) or impossible to deliver
  in time are dropped with a reason. If nothing remains, the Orchestrator sends a polite decline.
- **Bulk discount tiers** (by total units in the order, starting proposal):
  under 1,000 → 0%; 1,000–4,999 → 5%; 5,000–9,999 → 10%; 10,000 and above → 15%.
