#https://leetcode.com/problems/subarray-sum-equals-k/discuss/341399/Python-clear-explanation-with-code-and-example
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
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

#https://leetcode.com/problems/validate-binary-search-tree/discuss/32153/Python-version-based-on-inorder-traversal
#https://leetcode.com/problems/validate-binary-search-tree/discuss/974147/Python-O(n)-by-DFS-and-rule-w-Hint
#https://www.youtube.com/watch?v=s6ATEkipzow
#https://leetcode.com/problems/validate-binary-search-tree/discuss/146601/Python3-100-using-easy-recursion
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def valid(node,left,right):
            if not node:
                return True
            if not (node.val > left and node.val < right):
                return False
            return (valid(node.left,left,node.val) and valid(node.right,node.val,right))
        
        return valid(root,float("-inf"),float("inf"))


#https://www.youtube.com/watch?v=Ua0GhsJSlWM
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp = [[0 for i in range(len(text2)+1)] for j in range(len(text1)+1)]
        for i in range(len(text1)-1,-1,-1):
            for j in range(len(text2)-1,-1,-1):
                if text1[i]==text2[j]:
                    dp[i][j]= 1+ dp[i+1][j+1]
                else:
                    dp[i][j] = max(dp[i][j+1],dp[i+1][j])
        return dp[0][0]
    
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount+1]*(amount+1)
        dp[0] = 0
        for a in range(1,amount+1):
            for c in coins:
                if a-c >= 0:
                    dp[a] = min(dp[a],1+dp[a-c])
        return dp[amount] if dp[amount] != amount+1 else -1

class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        def dfs(root): # return pair of values: [withroot,withoutroot]
            if root is None:
                return [0,0]
            leftpair = dfs(root.left)
            rightpair = dfs(root.right)
            withroot = root.val + leftpair[1] + rightpair[1]
            withoutroot = max(leftpair) + max(rightpair)
            return [withroot,withoutroot]
        return max(dfs(root))
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        row = [1]*n
        for i in range(m-1):
            newRow = [1]*n
            for j in range(n-2,-1,-1):
                newRow[j] = newRow[j+1] + row[j]
            row = newRow
        return row[0]

class Solution:
    def longestPalindrome(self, str: str) -> str:
        
        pal = ""
        for i in range(len(str)):
            l,r = i,i #odd lengths
            found = self.findpal(str,l,r)
            if len(pal) <len(found):
                pal = found
            l,r = i,i+1
            found = self.findpal(str,l,r)
            if len(pal) <len(found):
                pal = found
        return pal
    def findpal(self,s,l,r):
        ls = len(s)
        while l>=0 and r<ls and s[l]==s[r]:
            l-=1
            r+=1
        return s[l+1:r]

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
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

#https://www.youtube.com/watch?v=wiGpQwVHdE0
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        leftPointer = 0
        result = 0
        charSet = set()
        for rightPointer in range(len(s)):
            while s[rightPointer] in charSet:
                charSet.remove(s[leftPointer]) # remove left char from window set
                leftPointer += 1
            charSet.add(s[rightPointer])
            result = max(result,rightPointer-leftPointer+1)
        return result

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        '''hashMap = {}
        for num in nums:
            if num in hashMap:
                hashMap[num] += 1
            else:
                hashMap[num] = 1
        for key,value in hashMap.items():
            if value ==1:
                res = key
        return res'''
        single = 0
        for i in range(32):
            count = 0
            for num in nums:
                if num & (1<<i) == (1<<i): count += 1
            single |= (count%3) << i
            
        return single if single < (1<<31) else single - (1<<32)

#https://leetcode.com/problems/set-matrix-zeroes/discuss/1469077/Python-From-O(M%2BN)-space-to-O(1)-space-with-Picture-Clean-and-Concise
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
    
        m, n = len(matrix), len(matrix[0])
        
        zeroFirstRow = any(matrix[0][c] == 0 for c in range(n))
        zeroFirstCol = any(matrix[r][0] == 0 for r in range(m))
        
        for r in range(1, m):
            for c in range(1, n):
                if matrix[r][c] == 0: matrix[0][c] = matrix[r][0] = 0

        for r in range(1, m):
            for c in range(1, n):
                if matrix[r][0] == 0 or matrix[0][c] == 0: matrix[r][c] = 0
                    
        if zeroFirstRow:
            for c in range(n): matrix[0][c] = 0
        
        if zeroFirstCol:
            for r in range(m): matrix[r][0] = 0

#https://www.youtube.com/watch?v=P6RZZMu_maU
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        numset = set(nums)
        longestSequence = 0
        for n in numset: 
            if (n-1) not in numset: # check if this num is starting of sequence
                length = 0
                while(n+length) in numset:
                    length += 1
                longestSequence = max(length,longestSequence)
        return longestSequence

class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        '''ans = []
        def perms(i, curr):
            if i >= len(s): # base case 
                ans.append(curr[:]) # reached of the character string s, now let's add the perms to the answer
                return 
            if s[i].isalpha():
                perms(i + 1, curr + s[i].lower()) # found a character, let's get all the lower case perms 
                perms(i + 1, curr + s[i].upper()) # found a character, let's get all the upper case perms 
            else:
                perms(i + 1, curr + s[i]) # found a number, just add it to the existing curr 
        perms(0, "") # starting index at 0 to get characters in string s and a current string set to ""
        return ans'''
        
        queue = []
        queue.append(s)
        n =  len(s)
        for i in range(n):
            c = s[i]
            if c.isalpha():
                size = len(queue)
                while size > 0:
                    s1 = queue.pop(0)
                    left = s1[0:i]
                    right = s1[i+1:]
                    queue.append(left+c.lower()+right)
                    queue.append(left+c.upper()+right)
                    size -= 1
        return queue

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        subset = []
        def dfs(i):
            if i >= len(nums):
                res.append(subset.copy())
                return
            # decision to include nums[i]
            subset.append(nums[i])
            dfs(i+1)
            #decision not to include nums[i]
            subset.pop()
            dfs(i+1)
        dfs(0)
        return res
    #https://leetcode.com/problems/subsets/discuss/973667/Backtracking-Template-or-Explanation-%2B-Visual-or-Python

#https://www.youtube.com/watch?v=Vn2v6ajA7U0
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        subset = []
        nums.sort()
        def dfs(i):
            if i >= len(nums):
                res.append(subset[::])
                return
            # decision to include nums[i]
            subset.append(nums[i])
            dfs(i+1)
            #decision not to include nums[i]
            subset.pop()
            while i+1< len(nums) and nums[i]==nums[i+1]: ## skip duplicates
                i += 1
            dfs(i+1)
        dfs(0)
        return res

#https://www.youtube.com/watch?v=pfiQ_PS1g8E&t=10s
#https://leetcode.com/problems/word-search/discuss/27660/Python-dfs-solution-with-comments.
#https://leetcode.com/problems/word-search/discuss/27820/Python-DFS-solution
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        row,col=len(board),len(board[0])
        path = set()
        def dfs(r,c,i):
            if i==len(word):
                return True
            if (r<0 or c<0 or r>=row or c>=col
                or word[i]!=board[r][c] or (r,c) in path):
                return False
            path.add((r,c))
            res =  (dfs(r+1,c,i+1) or
                    dfs(r-1,c,i+1) or
                    dfs(r,c+1,i+1) or
                    dfs(r,c-1,i+1))
            path.remove((r,c))
            return res
        for r in range(row):
            for c in range(col):
                if dfs(r,c,0): return True
        return False

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        countNums = {}
        freMap = [[] for i in range(len(nums)+1)]
        for n in nums:
            countNums[n] = 1+ countNums.get(n,0)
        for n,c in countNums.items():
            freMap[c].append(n)
        res = []
        for i in range(len(freMap)-1,0,-1):
            for n in freMap[i]:
                res.append(n)
                if len(res)==k:
                    return res

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        queue = [root]
        final_list = []
        if root is None:
            return final_list
        while(len(queue)>0):
            temp_list = [] # for even level
            for i in range(len(queue)):
                node = queue.pop(0)
                temp_list.append(node.val)
                if node.left is not None: queue.append(node.left)
                if node.right is not None: queue.append(node.right)
            final_list.append(temp_list)
        return sum(final_list[-1])

def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        
        N = len(grid)       # Height
        M = len(grid[0])    # Width
        
        # Transpose grid: Interchange rows and columns
        grid_t = zip(*grid)
                
        # Vertical and horizontal skylines
        sk_v = [max(row) for row in grid]     # As seen from left/right
        sk_h = [max(row) for row in grid_t]   # As seen from top/bottom
        
        res = 0
        for i in range(N):      # Rows of original grid
            for j in range(M):  # Columns of original grid
                # The new building cannot be higher than either skylines
                diff = min(sk_h[j], sk_v[i]) - grid[i][j]
                res += diff
        return res