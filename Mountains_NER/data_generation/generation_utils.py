def calc_usage_cost(usage):
    input_price_per_million = 0.75
    output_price_per_million = 4.50

    input_cost = (usage.prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.completion_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_costs(usages):
    total_cost = 0.0

    for usage in usages:
        cost = calc_usage_cost(usage)
        total_cost = total_cost + cost["total_cost"]

    return total_cost