from collections import deque
from email import charset
import heapq
import queue
import sys
from traceback import print_tb
from typing import Counter
from unittest import result

def targetSumSubset(numList,target):
    def helper(numList,idx,set,sos,target):
        if idx==len(numList):
            if sos==target:
                print(set+".")
            return

        helper(numList,idx+1,str(set)+str(numList[idx]),str(sos)+str(numList[idx]),target)
        helper(numList,idx,set,sos,target)
    helper(numList,0,"",0,target)
def targetSumSubset(numList,target):
    dp = ([[False for i in range(len(numList) + 1)]for i in range(target + 1)])
    for i in range(len(numList)+1):
        for j in range(target+1):
            if i==0 and j==0:
                dp[i][j] = True
            elif i==0:
                dp[i][j]=False
            elif j==0:
                dp[i][j]=True
            else:
                if dp[i-1][j]==True:
                    dp[i][j]=True
                else:
                    if dp[i-1][j-numList[i-1]]==True:
                        dp[i][j] = True
    return[numList][target]
def twoSum(numList,target):
    hashmap={} # val:index
    #result = []
    for i,n in enumerate(numList):
        diff = target-n
        if diff in hashmap:
            return [hashmap[diff],i]
        else:
            hashmap[n]=i
    return 
## Max Product Subarray
def maxProductSubarray(numList):
    ans = -134040
    currentProd = 1
    for i in range(len(numList)):
        currentProd *= numList[i]
        ans = max(ans,currentProd)
        if currentProd==0:
            currentProd = 1
    currentProd = 1
    for i in range(len(numList)-1,0,-1):
        currentProd *= numList[i]
        ans = max(ans,currentProd)
        if currentProd==0:
            currentProd = 1
    return ans
## Kadane's Algorithm for Maximum Sum Subarray
def maxSumSubarray(numlist):
    currentSum = numlist[0]
    bestSum = numlist[0]
    for i in range(1,len(numlist)):
        if currentSum>=0:
            currentSum += numlist[i]
        else:
            currentSum=numlist[i]
        if currentSum > bestSum:
            bestSum = currentSum
    return bestSum
## Maximum Sum Subarray with at least Size K 
def maxSumSubarrayK(numList,k):
    ans = -109988598735
    currentSum = numList[0]
    maxSum = [False]*len(numList)
    maxSum[0] = currentSum # store maxsubarray ending at every point
    for i in range(1,len(numList)):
        if currentSum>0:
            currentSum += numList[i]
        else:
            currentSum= numList[i]
        maxSum[i]=currentSum
    exactk = 0
    for i in range(k):
        exactk += numList[i]
    if exactk > ans: ans = exactk
    for i in range(k,len(numList)):
        exactk = numList[i] - numList[i-k]
        if exactk > ans: ans=exactk
        morethank =  maxSum[i-k] + exactk
        if morethank > ans: ans=morethank
    return ans
### Subarray Sum Equals K | Count Subarrays with Sum Equals K | Hashmap
def subarraySumK(numList,k):
    hashmap = {0:1} 
    curr_sum = 0
    ans = 0
    for i in range(len(numList)):
        curr_sum += numList[i]
        rsum = curr_sum -k
        if rsum in hashmap:
            ans += hashmap[rsum]
            
        elif rsum not in hashmap: 
            hashmap[curr_sum]=1
        
        else: hashmap[curr_sum] = hashmap[curr_sum]+1
    return ans
def subarraySum(nums,k):
		ans=0
		prefsum=0
		d={0:1}
		for num in nums:
			prefsum = prefsum + num

			if prefsum-k in d:
				ans = ans + d[prefsum-k]

			if prefsum not in d:
				d[prefsum] = 1
			else:
				d[prefsum] = d[prefsum]+1

		return ans
## Partioning an Array | Time and Space
def arrPartition(numList,target):
    def swap(arr,i,j):
        temp = arr[i]
        arr[i] = arr[j]
        arr[j] = temp
    i = j = 0
    while(i< len(numList)):
        if numList[i] > target:
            i = i+1
        else:
            swap(numList,i,j)
            i = i+1
            j = j+1
    return numList
def arrPartition01(numList,target):
    def swap(arr,i,j):
        temp = arr[i]
        arr[i] = arr[j]
        arr[j] = temp
    i = j = 0
    while(i< len(numList)):
        if numList[i] == target:
            i = i+1
        else:
            swap(numList,i,j)
            i = i+1
            j = j+1
    return numList
def arrPartition012(numList,target1,target2):
    def swap(arr,i,j):
        temp = arr[i]
        arr[i] = arr[j]
        arr[j] = temp
    i=j=k=0
    k = len(numList)-1
    while(i<=k):
        if numList[i] == target1:
            swap(numList,i,j)
            i = i+1
            j = j+1
        
        elif numList[i]==target2:
            i = i+1
        else:
            swap(numList,i,k)
            k = k - 1
    return numList
## Partition into Subsets
def partitionIntoSubsets(people,team):
    if people==0 or team==0 or people<team:
        return 0
    #dp = [[0]*(people+1)]*(team+1)
    dp = ([[0 for i in range(people+ 1)]for i in range(team + 1)])
    for t in range(1,team+1):
        for p in range(1,people+1):
            if p < t:
                dp[t][p]=0
            elif p==t:
                dp[t][p]=1
            else:
                dp[t][p] = dp[t-1][p-1] + dp[t][p-1]*t
    for i in dp:
        print(i)
    return dp[team][people]
## Best Time to Buy and Sell Stocks - One Transaction Allowed
def maxProfit(priceList):
    import sys
    least_so_far = -sys.maxsize # least price till current day
    overall_profit = 0 # overall profit
    today_sell_profit = 0 # if have to sell today, what will be the profit
    for i in range(len(priceList)):
        if priceList[i] < least_so_far:
            least_so_far = priceList[i]
        today_sell_profit = priceList[i]-least_so_far
        if today_sell_profit > overall_profit:
            overall_profit = today_sell_profit
    return overall_profit
## buy sell- infinity transaction allowed.
## but cant do buy-buy-sell-sell
## only can do buy-sell-buy-sell...
def maxProfitinfinity(priceList):
    buy_date = 0  ## index are the dates, for date,price in enumerate(price)
    sell_date = 0
    profit = 0
    for i in range(1,len(priceList)):
        if priceList[i] >= priceList[i-1]:
            sell_date = sell_date + 1
        else:
            profit +=  priceList[sell_date] - priceList[buy_date]
            sell_date = buy_date = i
    profit +=  priceList[sell_date] - priceList[buy_date] # last proft collect when there is no dip
    return profit
##Best Time to Buy and Sell Stocks with Transaction Fee and Infinite Transactions
## but cant do buy-buy-sell-sell
## only can do buy-sell-buy-sell...
def maxProfitinfinitywithFees(priceList,fees):
    """
    how much profit in bought state
    how much profit in sold state
    """
    obsp = -priceList[0] # old bought state profit
    ossp =  0 # old sold state profit
    for i in range(1,len(priceList)):
        nbsp = 0 # new buy state profit
        nssp = 0 # new sell state profit
        if ossp-priceList[i] > obsp:
            nbsp = ossp-priceList[i]
        else:
            nbsp = obsp
        if obsp + priceList[i] - fees > ossp:
            nssp = obsp + priceList[i] - fees
        else:
            nssp = ossp
        obsp = nbsp
        ossp = nssp
    return ossp
#Best Time to Buy and Sell Stocks with Cool down
# one day cool down. after selling again buying wait one day
def maxProfitCoolDown(price_list):
    obsp = -price_list[0]
    ossp = 0
    ocsp = 0
    for i in range(1,len(price_list)):
        nbsp = 0
        nssp = 0
        ncsp = 0
        if (ocsp-price_list[i]>obsp):
            nbsp = ocsp-price_list[i]
        else:
            nbsp =  obsp
        if obsp + price_list[i] > ossp:
            nssp = obsp + price_list[i]
        else:
            nssp = ossp
        if ossp > ocsp:
            ncsp = ossp
        else:
            ncsp = ocsp
        obsp = nbsp
        ossp = nssp
        ocsp = ncsp 
    return ossp
## Best Time to Buy and Sell Stocks - Two Transaction Allowed (Hard)
## till current date selling max profit,max(prevoius day selling profit[],current day selling profit)
def maxProfittofro(priceList):
    import sys
    mpist = 0 # max profit if sold today 
    dpmpisut = [0]*(len(priceList)) # store max profit which sold till current day
    least_so_far = priceList[0] # least price till current day
    overall_profit = 0
    for i in range(1,len(priceList)):
        if priceList[i] < least_so_far:
            least_so_far = priceList[i]
        mpist = priceList[i]-least_so_far
        if mpist > dpmpisut[i-1]:
            dpmpisut[i]=mpist
        else:
            dpmpisut[i]=dpmpisut[i-1]
    mpibt = 0 # max profit it bought today
    maxat = priceList[len(priceList)-1]# max after today
    dpmpibat = [0]*len(priceList) # max profit if bought today or after today
    for i in range(len(priceList)-1,0,-1):
        if priceList[i-1] > maxat:
            maxat = priceList[i-1]
        mpibt = maxat  - priceList[i-1]
        if mpibt > dpmpibat[i]:
            dpmpibat[i-i] = mpibt
        else:
            dpmpibat[i-1] = dpmpibat[i]
    for i in range(len(priceList)):
        if dpmpisut[i]+dpmpibat[i] > overall_profit:
            overall_profit = dpmpisut[i]+dpmpibat[i]
    return overall_profit
### minimun railway stations require 
def minStation(arriveTime,departure_time):
    currentPlatform = 1
    minPlatform = 1
    arriveTime.sort()
    departure_time.sort()
    i=1 # for arrival
    j=0  # for departure
    while (i < len(arriveTime) and j < len(arriveTime)):
        if (arriveTime[i] <= departure_time[j]):
            currentPlatform += 1
            i += 1
        elif (arriveTime[i] > departure_time[j]):
            currentPlatform -= 1
            j += 1
        if (currentPlatform > minPlatform):
            minPlatform = currentPlatform
    return minPlatform
## ### meeting room, find if a man can join all meeting or not
def canAttendMeetings(intervalList):
    sortedList = sorted(intervalList, key=lambda x: x[0])
    print(sortedList)
    for i in range(1,len(intervalList)):
        if sortedList[i-1][1] >= sortedList[i][0]:
            return False
            break
        #else:
        #    return False
    return True
## Require minimum number of meeting rooms

def minMeetingRoom(intervalList):
    """
    idea: sort starting and ending time
          and then compare starting and ending time
    """
    startTime = sorted(i[0] for i in intervalList)
    endTime = sorted(i[1] for i in intervalList)
    print(startTime,endTime)
    finalRoom = 1
    currentRoom = 1
    i=0
    j=0
    while i < len(startTime) and j < len(endTime):
        if startTime[i] > endTime[j]:
            currentRoom += 1
            i += 1
        else:
            currentRoom -= 1
            j += 1
        if currentRoom > finalRoom:
            finalRoom =  currentRoom
    return finalRoom
## Task scheduler
def taskScheduler(taskList,uTime):
    """
    Each time takes 1 unit of time
    find minimum units of time to complete all task
    Idea : first count the frequency of all tasks
            then start from higher frequency taks first
            then 
    """
    count =  Counter(taskList)
    print(count)
    maxHeap = [-cnt for cnt in count.values()]
    heapq.heapify(maxHeap)
    time = 0
    queue = deque()
    while maxHeap or queue: ## till maxheap or queue not empty
        time += 1   # increase time every time
        if maxHeap:  # tille maxheap not empty
            cnt =  heapq.heappop(maxHeap) + 1 ## pop from heap and add one
            if cnt !=0: # if not zero
                queue.append([cnt,time + uTime]) ## add into queue count and time, when next time this task has to be pop
            if queue and queue[0][1] == time: ## queue not empty and  task time match with time
                heapq.heappush(maxHeap,queue.popleft()[0])  # push to maxheap
    return time

### LRU cache implementation
#def lruCache():
## Coin Change Combination
def coinChangeCombination(coinsList,target):
    dp = [0]*(target+1)
    dp[0] = 1
    for i in range(len(coinsList)): ## this make combination
        j=coinsList[i]
        while j < len(dp):
            dp[j] += dp[j-coinsList[i]]
            j += 1
    return dp[target]
## Coin Change Permutations
def coinChangePermutation(coinList,target):
    dp = [0]*(target+1)
    dp[0] = 1
    for i in range(1,target+1):
        for coin in coinList:
            if coin <= i:
                #ramt =  i - coin # remaining amount
                dp[i] += dp[i - coin]  # dp[i - coin] number of ways to pay remaining amount
    return dp[target] 
## Longest Common Subsequence
def longestCommonSubsequence(string1,string2):
    """
    string1 = c1r1 # character1 + remaining string
    string2 = c2r2 # character2 + remaining string
    formula = c1r2Xc2r2 = -s(r1)  -s(r2)
                          c1s(r1)  c2s(r2)
    lcs(s1,s1) = 1 + lcs(r1,r2), c1=c2
                 max(lcs(r1,s2),lcs(s1,r2)), c1!=c2
    """
    dp = ([[0 for i in range(len(string1) + 1)] for i in range(len(string2) + 1)])

    for i in range(len(string1)-1,-1,-1):
        for j in range(len(string2)-1,-1,-1):
            c1 = string1[i]
            c2 = string2[j]
            #print(c1,c2)
            if c1 == c2:
                dp[i][j] =  1 + dp[i+1][j+1]
            else:
                dp[i][j] = max(dp[i+1][j],dp[i][j+1])
    for i in dp:
        print(i)
    return dp[0][0]
## Longest Common Substring 
def longestCommonSubstring(string1,string2):
    """
    Idea: Compare all prefixes of string1 with all prefixes of string2 
          and then find longes common suffix
          :: in other words... longest common ending part of both strings
          string1 =  r1c1
          string2 = r2c2
    """
    res = 0
    dp = ([[0 for i in range(len(string1) + 1)] for i in range(len(string2) + 1)])
    for i in range(1,len(string1)):
        for j in range(1,len(string2)):
            c1=string1[i-1]
            c2=string2[j-1]
            if c1==c2:
                dp[i][j] = dp[i-1][j-1]+1
            if dp[i][j]>res:
                res=dp[i][j]
    for i in dp:
        print(i)
    return res
## Count Palindromic Substrings
def countPalindromicSubstring(string):
    """
    Idea: c1+p+c2(e1=extreme,p=palindromic part,e2=extreme)
          if extremes are equal and middle part is palindrom then string will be palindrom
    """
    dp = ([[0 for i in range(len(string) + 1)] for i in range(len(string) + 1)])
    count = 0
    for gap in range(len(string)):
        i = 0 # dialonal start from 0th row and gap like 0,1,2,...
        j =  gap #gap + i # column start 
        while j < len(dp)-1: ## will go till last column bcoz every diagonal ending at last column and starting from first row
            if gap==0:
                dp[i][j]=1
            elif gap == 1:
                if string[i]==string[j]:
                    dp[i][j] = 1
            else:
                if string[i]==string[j] and dp[i+1][j-1] == 1:
                    dp[i][j] = 1
            if dp[i][j] == 1:
                count += 1
            j += 1
            i += 1
    for i in dp:
        print(i)
    return count

## Palindrome Partitioning with Minimum Cuts
def minPalindromicCut(string):
    """
    string =  abccbc... a|b|c|c|b|c -> 5 cuts
                        a|b|cc|b|c -> 4 cuts
                        a|bccb|c -> 2 cuts
            a|bccbc(left|right)
    what will be the min cuts to make every parts should be palindromic
    """
    dp = ([[False for i in range(len(string))] for i in range(len(string))])
    '''for gap in range(len(string)):
        i = 0 # dialonal start from 0th row and gap like 0,1,2,...
        j =  gap #gap + i # column start 
        while j < len(dp)-1: ## will go till last column bcoz every diagonal ending at last column and starting from first row
            print(i,j)
            if gap==0:
                dp[i][j]=0
            elif gap == 1:
                if string[i]==string[j]:
                    dp[i][j] = 0
                else:
                    dp[i][j] = 1
            else:
                if dp[i][j]:
                    dp[i][j] = 0
                else:
                    mcut = 294872 
                    for k in range(i,j):
                        leftPart = dp[i][k]
                        rightPart = dp[k+1][j]
                        if leftPart + rightPart + 1 < mcut:
                            mcut = leftPart + rightPart + 1
                    dp[i][j]=mcut
            j += 1
            i += 1'''
    for gap in range(len(string)):
        i = 0 # dialonal start from 0th row and gap like 0,1,2,...
        j =  gap #gap + i # column start 
        while j < len(dp): ## will go till last column bcoz every diagonal ending at last column and starting from first row
            if gap==0:
                dp[i][j]=True
            elif gap == 1:
                if string[i]==string[j]:
                    dp[i][j] = True
            else:
                if string[i]==string[j] and dp[i+1][j-1] == True:
                    dp[i][j] = True
            j += 1
            i += 1
    import sys
    dp_new = [0]*len(string)
    for j in range(1,len(dp)):
        mcut = sys.maxsize
        i = j
        while i >= 1:
        #for i in range(j-1,-1,-1):
            #print(j,i)
            if dp[i][j]==True:
                #print(dp_new[i-1])
                if dp_new[i-1] < mcut:
                    mcut = dp_new[i-1]
            i -= 1
        dp_new[j] = mcut+1
    #for i in dp_new:
    #    print(i)
    print(dp_new)
    return dp_new[len(string)-1]
## Minimum Window Substring 
def minWindowsubstring(string1,string2):
    hashMap_string2 = {}
    for i in range(len(string2)):
        c = string2[i]
        if c in hashMap_string2:
            hashMap_string2[c] += 1
        else:
            hashMap_string2[c] = 1
    print(hashMap_string2)
    hashMap_string1 = {}
    mcount = 0 # match count
    dmcount = len(string2) # desire match count
    i = -1
    j = -1
    while( i < len(string1)-1 and mcount < dmcount):
        i += 1
        ch = string1[i]
        if ch in hashMap_string1:
            hashMap_string1[ch] += 1
        else:
            hashMap_string1[ch] = 1
        if hashMap_string1[ch] <= hashMap_string2[ch]: ## if frequency of ch in map2 is less or equal then only increase match count 
            mcount += 1
## Climbing Stairs
def climbStairs(n):
    """
    dp = [0]*(n+1)
    dp[0] = 1
    for i in range(1,n+1):
        if i==1:
            dp[i]=dp[i-1]
        elif i==2:
            dp[i]=dp[i-1]+dp[i-2]
        else:
            dp[i] =  dp[i-1]+dp[i-2]+dp[i-3]
    for i in dp:
        print(i)
    return dp[n]
    """
    #Climbing Stairs with Jumps
    """dp = [0]*(len(n)+1)
    dp[len(n)] = 1
    for i in range(len(n)-2,-1,-1):
        for j in range(1,n[i]+1):
            if j+i < len(dp):
                dp[i] += dp[i+j]
    #for i in dp:
    #    print(i)
    print(dp)
    return dp[0]"""
    ##Climbing Stairs with Minimum Moves/jumps
    dp = [None]*(len(n)+1)
    dp[len(n)] = 0
    for i in range(len(n)-2,-1,-1):
        if n[i]==0:
            pass
        else:
            mini = sys.maxsize
            for j in range(1,n[i]+1):
                if j+i < len(dp):
                    if dp[i+j]!=None:
                        mini = min(mini,dp[i+j])
            if mini != sys.maxsize:
                dp[i] = mini+1
            else:pass
            
    print(dp)
    return dp[0]
## Minimum Cost Path
def minCostPath(costMatrix):
    dp = ([[0 for i in range(len(costMatrix[0]))] for i in range(len(costMatrix))])
    #for i in costMatrix:
    #    print(i)
    #for i in dp:
    #    print(i)
    for i in range(len(dp)-1,-1,-1):
        for j in range(len(dp[0])-1,-1,-1):
            if i==len(dp)-1 and j==len(dp[0])-1:
                dp[i][j]=costMatrix[i][j]
            elif i==len(dp)-1:
                dp[i][j]=dp[i][j+1]+ costMatrix[i][j]
            elif j==len(dp[0])-1:
                dp[i][j] = dp[i+1][j] + costMatrix[i][j]
            else:
                dp[i][j] = min(dp[i+1][j],dp[i][j+1]) + costMatrix[i][j]
    for i in dp:
        print(i)
    return dp[0][0]
## 0-1 Knapsack Problem 
def knapsack01(weights,values,capacity):
    dp = ([[0 for i in range(capacity + 1)] for i in range(len(weights) + 1)])
    print(len(dp),len(dp[0]))
    #for i in dp:
    #    print(i)
    for i in range(1,len(weights)+1):
        for j in range(1,capacity+1):        
            if j >= weights[i-1]: # when remaining weights greater than 0 
                if dp[i-1][j-weights[i-1]] + values[i-1] > dp[i-1][j]: # j-weights[i-1]- remaining capacity
                    dp[i][j] = dp[i-1][j-weights[i-1]] + values[i-1]
                else:
                    dp[i][j] = dp[i-1][j] # when j doesnt participate
    for i in dp:
        print(i)
    return dp[len(weights)][capacity]

### Path with Maximum Gold
def getMaxGold(goldMatrix):
    ## when start from column 1st and any row
    '''dp = ([[0 for i in range(len(goldMatrix[0]))] for i in range(len(goldMatrix))])
    #print(len(goldMatrix))
    #for i in goldMatrix:
    #    print(i)
    #print("\n")
    for col in range(len(goldMatrix[0])-1,-1,-1):
        for row in range(len(goldMatrix)):
            if col==len(goldMatrix[0])-1:
                dp[row][col] = goldMatrix[row][col]
            elif row==len(goldMatrix)-1:
                dp[row][col] = max(dp[row][col+1],dp[row-1][col+1]) + goldMatrix[row][col]
            elif row==0:
                dp[row][col] = max(dp[row][col+1],dp[row+1][col+1]) + goldMatrix[row][col]
            else:
                dp[row][col] = max(dp[row][col+1],dp[row+1][col+1],dp[row-1][col+1]) + goldMatrix[row][col]
    for i in dp:
        print(i)
    return max([row[0] for row in dp] )'''
    ## can start from anywhere in the matrix
    def dfs(i, j, gold):
            # check that current position is in grid and has gold
            if i < 0 or i >= len(goldMatrix) or j < 0 or j >= len(goldMatrix[i]) or goldMatrix[i][j] == 0:
                return gold
			# do not check a position that was already visted during the current DFS
            if goldMatrix[i][j] == "#":
                return gold
			
			# add the gold at this position to the current total
            gold += goldMatrix[i][j]
			
			# mark this position as visited
            temp = goldMatrix[i][j]
            goldMatrix[i][j] = "#"

			# run DFS on all the neighbors and take the max amount of gold 
            gold = max(dfs(i+1, j, gold), dfs(i-1, j, gold), dfs(i, j+1, gold), dfs(i, j-1, gold))

			# reset the curr grid position gold amount so it may be visited from different DFS paths
            goldMatrix[i][j] = temp

            return gold
    maxGold = 0
	# loop through each grid space and check if it has gold, if so, run DFS
    for i in range(len(goldMatrix)):
        for j in range(len(goldMatrix[i])):
            if goldMatrix[i][j] == 0:
                continue
			    # keep track of which DFS produced the max amount of gold
            maxGold = max(maxGold, dfs(i, j, 0))
    return maxGold
## Count Binary Strings
##def countBinaryString():
## Minimum Domino Rotations for Equal Row
##Count Binary Strings
#def equalSumSubset(arr):
## divide n poeple in k partitions
##def kPartition(arr):
## Longest Substring Without Repeating Characters
def longestSWRC(string):
    """
    use sliding window approach
    left pointer and right pointer
    """
    leftPointer = 0
    result = 0
    charSet = set()
    for rightPointer in range(len(string)):
        while string[rightPointer] in charSet:
            charSet.remove(string[leftPointer])
            leftPointer += 1
        charSet.add(string[rightPointer])
        result = max(result,rightPointer-leftPointer+1)
    return result
## Longest Substring With Exactly K Distinct Characters
#def longestSWEKDC(str):
def permute( nums):
        if len(nums) == 1:
            return [nums]
        result = []
        for i in range(len(nums)):
            others = nums[:i] + nums[i+1:]
            #print(others)
            other_permutations = permute(others)
            print(other_permutations)
            for permutation in other_permutations:
                result.append([nums[i]] + permutation)
        return result 
# Kth Largest Element in an Array
def findKthLargest(nums, k):
        minHeap = []
        for x in nums:
            heapq.heappush(minHeap, x)
            print(minHeap)
            if len(minHeap) > k:
                heapq.heappop(minHeap)
        return minHeap[0]
def letterCombinations(D):
        L = {'2':"abc",'3':"def",'4':"ghi",'5':"jkl",
                '6':"mno",'7':"pqrs",'8':"tuv",'9':"wxyz"}
        lenD, ans = len(D), []
        if D == "": return []
        def dfs(pos: int, st: str):
            #print(pos,ans)
            if pos == lenD: ans.append(st)
            else:
                letters = L[D[pos]]
                for letter in letters:
                    print("ok")
                    dfs(pos+1,st+letter)
        dfs(0,"")
        return ans
def maxProduct( A):
        B = A[::-1] # reverse the list
        print(B)
        for i in range(1, len(A)):
            A[i] *= A[i - 1] or 1
            B[i] *= B[i - 1] or 1
        print(A,B)
        return max(A + B)
def letterCasePermutation(string):
    queue = []
    queue.append(string)
    n =  len(string)
    for i in range(n):
        c = string[i]
        if c.isalpha():
            size = len(queue)
            while size > 0:
                s = queue.pop(0)
                left = s[0:i]
                right = s[i+1:]
                queue.append(left+c.lower()+right)
                queue.append(left+c.upper()+right)
                size -= 1
    return queue


#print(targetSumSubset([10,20,30,40],70))
#print(twoSum([10,20,30,40],70))
#print(maxProductSubarray([1,4,-1,5,6,2]))
#print(maxSumSubarray([1,4,-1,5,6,2]))
#print(maxSumSubarrayK([1, 2, 3, -10, -3],4))
#print(subarraySumK([1, 1, 1],2))
#print(arrPartition([9,4,5,1,6,7,2],5))
#print(arrPartition01([1,1,0,1,0,0,1,1,1,0,0,1],1))
#print(arrPartition012([1,1,0,2,1,0,2,0,1,1,2,2,1,0,0,2,2,1,2],0,1))
#print(partitionIntoSubsets(5,4))
#print(maxProfit([-1,23,2,5,27,6,2,56,7,2,1,78]))
#print(maxProfitinfinity([1,2,3]))
#print(maxProfitinfinitywithFees([10,20,30],2))
#print(maxProfitCoolDown([10,20,30]))
#print(maxProfittofro([3,3,5,0,0,3,1,4]))
#print(canAttendMeetings([[1,2],[2,4],[5,9],[4,7]]))
#print(minMeetingRoom([[1,2],[2,4],[5,9],[4,7]]))
#print(taskScheduler(["A","A","A","B","B","C","C"],1))
#print(coinChangeCombination([2,3,5,6],10))
#print(coinChangePermutation([2,3,5,6],10))
#print(longestCommonSubsequence("abcd","aebd"))
#print(longestCommonSubstring("pqabcxp","xyzabcp"))
#print(countPalindromicSubstring("aaa"))#"abccbc"))
#print(minPalindromicCut("abccbc"))
#print(minWindowsubstring("abcbjdadbajdbaerqubahd","abcdc")) # incomplete yet
#print(climbStairs([3,2,4,2,0,2,3,1,2,2]))
#print(minCostPath([ [ 1, 2, 3 ],[ 4, 8, 2 ],[ 1, 5, 3 ] ]))#([[1,0,2,3],[9,6,2,5],[4,2,6,8],[9,9,0,4],[2,1,5,1]]))
#print(knapsack01([2,5,1,3,4],[15,14,10,45,30],7))
#print(getMaxGold([[1,0,7],[2,0,6],[3,4,5],[0,3,0],[9,0,20]]))#[[0,6,0],[5,8,7],[0,9,0]]))#[ [ 1, 2, 0, 7 ],[ 4, 8, 2, 0 ],[ 1, 5, 3, 5 ] ]))
#print(longestSWRC("abcabcbb"))
#print(permute([1,2,3]))
#print(findKthLargest([3,2,1,5,6,4],2))
#print(letterCombinations("23"))
#print(maxProduct([2,3,-2,4]))
print(letterCasePermutation("a1b2"))

"""
driver code
def main():
    n = int(input())
    arr = []
    for i in range(0, n) :
        arr.append(int(input()))

    dp = [0] * (n + 1)

    print(fun(n, arr, dp))

if __name__ == "__main__":
    main()
"""