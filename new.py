'''s1 = "abcd"
s2 = "acbd"
dp = ([[0 for i in range(len(s1) + 1)]
                for i in range(len(s2) + 1)])
for i in range(len(s1)-1,-1,-1):
    for j in range(len(s2)-1,-1,-1):
        c1=s1[i]
        c2=s2[j]
        if c1==c2:
            dp[i][j] = dp[i+1][j+1]+1
        else:
            dp[i][j] = max(dp[i+1][j],dp[i][j+1])
for i in dp:
    print(i)


def csp(n):
    dp = [0]*(n+1)
    dp[0]=1
    for i in range(1,n+1):
        if i==1:
            dp[i]=dp[i-1]
        elif i==2:
            dp[i]=dp[i-1][i-2]
        else:
            dp[i]=dp[i-1]+dp[i-2]+dp[i-3]
    return dp
res = csp(10)
for i in res:
    print(i)'''
    #sentence embedding ad words embedding
l = [5,6,3,7]
#for i in range(len(l)-1,-1,-1):
#    print(l[i])
def containsDuplicate(nums) -> bool:
        hashMap = {}
        flag = False
        for i in nums:
            print(hashMap)
            if i in hashMap:
                hashMap[i] += 1
                val = hashMap[i]
                if val >=2:
                    #print(val)
                    flag =  True
                    break
            else:
                hashMap[i] = 1
        return flag
print(containsDuplicate([1,2,3,1]))

    
