$inPath = "C:\Users\italo\Documents\Ciclo 2026-2\Deep Learning\TA\Entregable_2\presentacion_waimlap\Poster_DINOv2_EuroSAT_LoRA_Mejorado.pptx"
$outPath1 = "C:\Users\italo\Documents\Ciclo 2026-2\Deep Learning\TA\Entregable_2\presentacion_waimlap\Poster_DINOv2_EuroSAT_LoRA_Mejorado.pdf"
$outPath2 = "C:\Users\italo\Documents\Ciclo 2026-2\Deep Learning\TA\Entregable_2\Poster_DINOv2_EuroSAT_LoRA_Mejorado.pdf"
$outPath3 = "C:\Users\italo\Documents\Ciclo 2026-2\Deep Learning\TA\Entregable_2\presentacion_waimlap\Poster_DINOv2_EuroSAT_LoRA.pdf"
$outPath4 = "C:\Users\italo\Documents\Ciclo 2026-2\Deep Learning\TA\Entregable_2\Poster_DINOv2_EuroSAT_LoRA.pdf"

$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open($inPath, [Microsoft.Office.Core.MsoTriState]::msoFalse, [Microsoft.Office.Core.MsoTriState]::msoFalse, [Microsoft.Office.Core.MsoTriState]::msoFalse)
$pres.SaveAs($outPath1, 32)
$pres.SaveAs($outPath2, 32)
$pres.SaveAs($outPath3, 32)
$pres.SaveAs($outPath4, 32)
$pres.Close()
$ppt.Quit()
Write-Host "Exported PDF successfully to all target paths"
