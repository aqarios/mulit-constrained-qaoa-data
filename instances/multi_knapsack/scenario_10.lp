Minimize
 obj: - 19 x_0_0 - 19 x_0_1 - 16 x_1_0 - 16 x_1_1 - 16 x_2_0 - 18 x_2_1 

Subject To 
 ohc_item_0: + 1 x_0_0 + 1 x_0_1 + 1 x_0_2  = 1.0
 ohc_item_1: + 1 x_1_0 + 1 x_1_1 + 1 x_1_2  = 1.0
 ohc_item_2: + 1 x_2_0 + 1 x_2_1 + 1 x_2_2  = 1.0
 knapsack_0: + 2 x_0_0 + 4 x_1_0 + 4 x_2_0  <= 11.0
 knapsack_1: + 2 x_0_1 + 4 x_1_1 + 4 x_2_1  <= 8.0

Bounds

Binary
 x_0_0 x_0_1 x_1_0 x_1_1 x_2_0 x_2_1 x_0_2 x_1_2 x_2_2
General

End