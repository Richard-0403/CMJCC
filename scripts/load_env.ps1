# 把 .env 导入当前 PowerShell 会话。
#
# 为什么需要它：应用**刻意不**自动加载 .env——密钥只从进程环境读取
# （src/jobrec/llm/remote_provider.py 的 API_KEY_ENV，R26.1），配置文件里
# 放不进密钥（AppConfig 禁止额外字段）。因此运行 remote hybrid 或 postgres
# 测试之前，必须先把变量导出到环境里。
#
# 为什么放在 scripts/ 而不是改 src/：experiment id 的源码指纹覆盖
# jobrec/ 与 jobrec_eval/ 下的全部 *.py。往 src/ 里加一个 load_dotenv 会改变
# 指纹，使已归档的 exp-f90573008bdb 与当前代码不再对应。放这里零影响。
#
# 用法（注意前面的点，必须 dot-source 才能影响当前会话）：
#   . .\scripts\load_env.ps1
#
# 之后同一个终端里就可以跑：
#   .venv\Scripts\pytest.exe -m postgres
#   .venv\Scripts\python.exe -m jobrec_eval.cli pipeline --config configs/hybrid_vectorengine.yaml ...

param([string]$Path = ".env")

if (-not (Test-Path $Path)) {
    Write-Error "$Path 不存在。请先从 .env.example 复制一份并填好。"
    return
}

$loaded = @()
foreach ($line in Get-Content $Path) {
    $t = $line.Trim()
    if ($t -eq "" -or $t.StartsWith("#") -or -not $t.Contains("=")) { continue }
    $i = $t.IndexOf("=")
    $k = $t.Substring(0, $i).Trim()
    $v = $t.Substring($i + 1).Trim()
    # 去掉可能存在的引号
    if ($v.Length -ge 2 -and (($v.StartsWith('"') -and $v.EndsWith('"')) -or ($v.StartsWith("'") -and $v.EndsWith("'")))) {
        $v = $v.Substring(1, $v.Length - 2)
    }
    Set-Item -Path "Env:$k" -Value $v
    $loaded += $k
}

# 只报变量名，绝不回显值——尤其是 *_API_KEY / *PASSWORD / *SECRET / *TOKEN。
Write-Host "已从 $Path 导入 $($loaded.Count) 个变量: $($loaded -join ', ')"
foreach ($k in $loaded) {
    if ($k -match 'KEY|PASSWORD|SECRET|TOKEN') {
        $len = (Get-Item "Env:$k").Value.Length
        Write-Host ("  {0,-22} <已设置, 长度 {1}, 值不回显>" -f $k, $len) -ForegroundColor DarkGray
    }
}
